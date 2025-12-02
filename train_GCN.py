import os
import sys
import math
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple
from torch_geometric.utils import add_self_loops  # NEW: 补自环
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils.link_utils import SynGraphDataset, split_edge,clear_time,clear_time_UCI
# Datasets
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

# ------------- Repro -------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ------------- Adj helpers -------------
def build_sparse_adj(edge_index: torch.Tensor, num_nodes: int, edge_weight: Optional[torch.Tensor] = None, device: str = "cpu"):
    # edge_index: [2, E]
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    indices = edge_index.to(device)
    values = edge_weight.to(device)
    adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
    return adj.coalesce()

def prepare_adj(edge_index: torch.Tensor, num_nodes: int, normalize_adj: bool, device: str):
    if normalize_adj:
        ei, ew = gcn_norm(edge_index=edge_index, edge_weight=None, num_nodes=num_nodes, add_self_loops=True, dtype=torch.float32)
        return build_sparse_adj(ei, num_nodes, ew, device)
    else:
        return build_sparse_adj(edge_index, num_nodes, None, device)

# ------------- Model (keep same as node_curvature_continuous) -------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if support.is_cuda and adj.is_cuda:
            pass
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = torch.mm(adj, support)
        return out

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def back(self, x, adj):
        x0 = self.gc1(x, adj)
        x1 = F.relu(x0)
        x2 = self.gc2(x1, adj)
        return (x0, x1, x2)

# ------------- IO utils -------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_model(task: str, dataset: str, model: nn.Module,normalize_adj: bool):
    ckpt_dir = "./checkpoints"
    ensure_dir(ckpt_dir)
    norm_flag = "norm" if normalize_adj else "nonorm"
    path = os.path.join(ckpt_dir, f"{task}_{dataset}_{norm_flag}.pt")
    torch.save({"task": task, "dataset": dataset, "state": model.state_dict()}, path)
    return path

# ------------- Tasks -------------
@dataclass
class TrainConfig:
    task: str
    dataset: str
    root: str = "./data"
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    hidden_dim: int = 64
    dropout: float = 0.5
    batch_size: int = 128  # for graph classification
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    eval_every: int = 1
    link_num_neg: int = 1  # negatives per positive
    # 类均衡划分（写死使用）
    train_ratio: float = 0.5
    val_ratio: float = 0.3
    # 可选邻接归一化
    normalize_adj: bool = False

# ---- Balanced split for node classification (always applied) ----
def apply_balanced_split_(data, train_ratio: float, val_ratio: float, seed: int):
    y_cpu = data.y.detach().cpu()
    num_nodes = y_cpu.size(0)
    labelslist = torch.unique(y_cpu).tolist()
    rng = random.Random(seed)
    node_labels = {int(c): [] for c in labelslist}
    for i, yi in enumerate(y_cpu.tolist()):
        node_labels[yi].append(i)

    idx_train, idx_val, idx_test = [], [], []
    for c in labelslist:
        rng.shuffle(node_labels[c])
        n = len(node_labels[c])
        n_train = math.floor(train_ratio * n)
        n_val = math.floor(val_ratio * n)
        idx_train += node_labels[c][:n_train]
        idx_val += node_labels[c][n_train:n_train + n_val]
        idx_test += node_labels[c][n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    data.train_mask = train_mask.to(data.x.device)
    data.val_mask = val_mask.to(data.x.device)
    data.test_mask = test_mask.to(data.x.device)


class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def initializeNodes(dataset):
    """为没有节点特征的数据集生成基于度数的特征"""
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

# ---- Node Classification ----
def run_node_classification(cfg: TrainConfig):
    dataset = Planetoid(root=os.path.join(cfg.root, "Planetoid"), name=cfg.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(cfg.device)

    # 始终应用类均衡划分
    apply_balanced_split_(data, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed)

    ei = data.edge_index
    if not cfg.normalize_adj:
        has_loop = (ei[0] == ei[1]).any().item()
        if not has_loop:
            ei, _ = add_self_loops(ei, num_nodes=data.num_nodes)

    adj = prepare_adj(ei, data.num_nodes, cfg.normalize_adj, cfg.device)

    print('adj',adj)

    model = GCN(nfeat=dataset.num_features, nhid=cfg.hidden_dim, nclass=dataset.num_classes, dropout=cfg.dropout).to(cfg.device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def evaluate(split: str):
        model.eval()
        with torch.no_grad():
            logits = model(data.x, adj)
            if split == "train":
                mask = data.train_mask
            elif split == "val":
                mask = data.val_mask
            else:
                mask = data.test_mask
            pred = logits.argmax(dim=-1)
            acc = (pred[mask] == data.y[mask]).float().mean().item()
        return acc

    best_val = 0.0
    best_test = 0.0
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, adj)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % cfg.eval_every == 0:
            train_acc = evaluate("train")
            val_acc = evaluate("val")
            test_acc = evaluate("test")
            print(f"[Node] Epoch {epoch:03d} | loss {loss:.4f} | train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f} ")

    last_path = save_model("node", cfg.dataset, model,cfg.normalize_adj)

class NetLinkTrain(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, add_self_loops: bool, normalize: bool):
        super(NetLinkTrain, self).__init__()
        # keep same settings: bias=False
        self.conv1 = GCNConv(in_dim, hidden, add_self_loops=add_self_loops, normalize=normalize, bias=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=add_self_loops, normalize=normalize, bias=False)
        self.linear = nn.Linear(hidden * 2, 2, bias=False)
        self.dropout = dropout

    def encode(self, x, edgeindex):
        z = self.conv1(x.to(torch.float32), edgeindex)
        z = z.relu()
        # 可选dropout，如果要完全对齐原文件，保持注释掉
        # z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.conv2(z, edgeindex)
        return z

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        logits = self.linear(h)  # 2-class logits
        return logits

def get_link_labels(pos_edge_index, neg_edge_index, device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.long, device=device)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

# ---- Link Prediction ----
def run_link_prediction(cfg: TrainConfig):
    device = cfg.device
    dataset_name = cfg.dataset  # 例如 'UCI'（建议使用你在 link_curvature 中的数据名）
    dataset_dir = os.path.join(cfg.root, dataset_name)
    dataset = SynGraphDataset(dataset_dir, dataset_name)
    data = dataset[0]
    data = train_test_split_edges(data)  # 生成 train/val/test 的 pos/neg 边
    data = data.to(device)

    time_dict = data.time_dict
    if dataset_name == 'UCI':
        clear_time_dict = clear_time_UCI(time_dict)
    else:
        clear_time_dict = clear_time(time_dict)

    edge_index_old = split_edge(0, 200, 'month', clear_time_dict, data.num_nodes)
    edgeindex = torch.tensor(edge_index_old, device=device)

    train_pos=edgeindex

    # 构图边：使用训练正边（与参考脚本一致思路）


    # 模型设置与参考脚本一致：是否自环/归一化由开关控制
    add_loops = cfg.normalize_adj
    use_norm = cfg.normalize_adj
    model = NetLinkTrain(in_dim=dataset.num_features, hidden=cfg.hidden_dim, dropout=cfg.dropout,
                         add_self_loops=add_loops, normalize=use_norm).to(cfg.device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def sample_neg(edge_index_pos, num_nodes, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = edge_index_pos.size(1) * cfg.link_num_neg
        neg = negative_sampling(
            edge_index=edge_index_pos, num_nodes=num_nodes,
            num_neg_samples=num_samples, force_undirected=True, method='sparse'
        ).to(cfg.device)
        return neg

    def evaluate(prefix: str):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, train_pos)
            if prefix == "train":
                # 训练集没有预定义的负边，需要动态采样
                pos = train_pos
                neg = sample_neg(train_pos, data.num_nodes, train_pos.size(1))
            else:
                pos = data[f"{prefix}_pos_edge_index"]
                neg = data[f"{prefix}_neg_edge_index"]
            # pos = data[f"{prefix}_pos_edge_index"]
            # neg = data[f"{prefix}_neg_edge_index"]
            logits = model.decode(z, pos, neg)  # [E,2]
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu()
            # 近似AUC（无sklearn）：排序近似
            sorted_idx = torch.argsort(probs)
            ranks = torch.zeros_like(sorted_idx)
            ranks[sorted_idx] = torch.arange(len(probs))
            pos_num = pos.size(1)
            neg_num = neg.size(1)
            pos_ranks = ranks[:pos_num]
            auc = (pos_ranks.float().mean() - (pos_num - 1) / 2) / max(neg_num, 1)
        return auc.item()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(data.x, train_pos)
        neg = sample_neg(train_pos, data.num_nodes, train_pos.size(1))
        logits = model.decode(z, train_pos, neg)
        labels = get_link_labels(train_pos, neg, cfg.device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if epoch % cfg.eval_every == 0:
            train_auc = evaluate("train")
            val_auc = evaluate("val")
            test_auc = evaluate("test")
            print(f"[Link] Epoch {epoch:03d} | loss {loss:.4f} | train AUC {train_auc:.4f} | val AUC {val_auc:.4f} | test AUC {test_auc:.4f}")

    save_model("link", cfg.dataset, model, cfg.normalize_adj)

# ---- Graph Classification ----
def run_graph_classification(cfg: TrainConfig):
    dataset = TUDataset(root=os.path.join(cfg.root, "TUDataset"), name=cfg.dataset, use_node_attr=True, transform=NormalizeFeatures())

    initializeNodes(dataset)

    if dataset.transform is not None:
        dataset = TUDataset(root=os.path.join(cfg.root, "TUDataset"), name=cfg.dataset,
                            use_node_attr=True, transform=dataset.transform)


    num_classes = dataset.num_classes
    in_dim = dataset.num_features

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    # 模型直接输出图分类的类别维度（与 node_curvature_continuous 的无 head 方式一致）
    model = GCN(nfeat=in_dim, nhid=cfg.hidden_dim, nclass=num_classes, dropout=cfg.dropout).to(cfg.device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def loop(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_num = 0

        for batch in loader:
            batch = batch.to(cfg.device)
            if train:
                opt.zero_grad()

            # 构造批图的邻接（注意：PyG 的 batch 没有 num_nodes 属性，使用 batch.x.size(0)）
            num_nodes = batch.x.size(0)
            if cfg.normalize_adj:
                ei, ew = gcn_norm(edge_index=batch.edge_index, edge_weight=None, num_nodes=num_nodes, add_self_loops=True, dtype=torch.float32)
                adj = build_sparse_adj(ei, num_nodes, ew, cfg.device)
            else:
                adj = build_sparse_adj(batch.edge_index, num_nodes, None, cfg.device)

            logits = model(batch.x, adj)
            # 无 head：对节点 logits 做图级平均池化
            graph_logits = global_mean_pool(logits, batch.batch)

            loss = F.cross_entropy(graph_logits, batch.y)

            if train:
                loss.backward()
                opt.step()

            total_loss += loss.item() * batch.num_graphs
            pred = graph_logits.argmax(dim=-1)
            total_correct += (pred == batch.y).sum().item()
            total_num += batch.num_graphs

        return total_loss / max(total_num, 1), total_correct / max(total_num, 1)

    best_val = 0.0
    best_test = 0.0
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = loop(train_loader, train=True)
        if epoch % cfg.eval_every == 0:
            val_loss, val_acc = loop(val_loader, train=False)
            test_loss, test_acc = loop(test_loader, train=False)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_path = save_model("graph", cfg.dataset, model, cfg.normalize_adj)
            print(f"[Graph] Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f} | test {test_loss:.4f}/{test_acc:.4f} | best(test@best_val) {best_test:.4f}")

    # last_path = save_model("graph", cfg.dataset, model, tag="last")
    if best_path is not None:
        print(f"[Graph] Saved model to {best_path}")
    else:
        last_path = save_model("graph", cfg.dataset, model, cfg.normalize_adj)
        print(f"[Graph] Saved model to {last_path}")

# ------------- CLI -------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Unified GCN training for node/link/graph tasks (no extra heads)")
    p.add_argument("--task", type=str, choices=["node_classification", "link_prediction", "graph_classification"],default='node_classification')
    p.add_argument("--dataset", type=str, help="Planetoid: Cora/CiteSeer/PubMed; TUDataset: e.g., PROTEINS, IMDB-BINARY",default='Cora')
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--link_num_neg", type=int, default=1)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.3)
    p.add_argument("--normalize_adj", action="store_true",default=True)
    args = p.parse_args()

    return TrainConfig(
        task=args.task,
        dataset=args.dataset,
        root=args.root,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        eval_every=args.eval_every,
        link_num_neg=args.link_num_neg,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        normalize_adj=args.normalize_adj
    )

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    torch.cuda.set_device(0) if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else None

    if cfg.task == "node_classification":
        run_node_classification(cfg)
    elif cfg.task == "link_prediction":
        run_link_prediction(cfg)
    elif cfg.task == "graph_classification":
        run_graph_classification(cfg)

if __name__ == "__main__":
    main()