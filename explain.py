import os
import sys
import copy
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from dig.xgraph.method import DeepLIFT, FlowMask, GNN_LRP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from enum import Enum

from utils.models import GCN_explainer
from utils.node_utils import (
    rumor_construct_adj_matrix, matrixtodict, clear, softmax,
    k_hop_subgraph, subadj_map, subfeaturs, rumor_construct_adj_matrix_v2,
    dfs2, reverse_paths, edge_percentage, custom_log_transform,normalize_ricci, from_edges_to_evaulate
)
from explainer_unified import UnifiedExplainer, ExplainerFactory

class RunMode(Enum):
    TRAIN = "train"      # 筛选训练节点
    VAL = "val"          # 筛选验证节点
    SELECT = "select"    # Optuna 优化 lambda

# ========== 配置类 ==========
@dataclass
class ExplainConfig:
    # 数据集配置
    dataset: str = "Cora"  # Cora, Citeseer, PubMed
    data_root: str = "./data"

    # 模型配置
    hidden_dim: int = 16
    dropout: float = 0.6
    layer_numbers: int = 2
    normalize_adj: bool = True

    # 解释方法
    method: str = "deeplift"  # deeplift, flowx, gnnexplainer, gnnlrp, pgexplainer
    run_mode: str = "select"

    # Ricci 曲率配置
    ricci_alpha: float = 0.0
    ricci_lambda: float = 0.0

    # 节点采样配置
    sample_num_nodes: int = 500
    train_ratio: float = 0.5
    percentage_threshold: float = 0.8

    n_trials: int = 50
    lambda_min: float = 0.0
    lambda_max: float = 0.1
    max_train_nodes: int = 250

    # 扰动配置
    changed_ratio: float = 0.5
    add_ratio: float = 0.5
    num_remove_val_ratio: float = 0.1
    num_add_val_ratio: float = 0.1

    # 保存路径
    result_root: str = "./result/"
    edge_mask_dir: str = "edge_masks_normalize"
    lambda_dir: str = "normalize/ricci"



    # 其他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ========== GCN 模型 (用于加载权重) ==========
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        return torch.mm(adj, support)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


# ========== 数据生成类 ==========
class GenData:
    def __init__(self, cfg: ExplainConfig):
        self.cfg = cfg
        self.changed_ratio = cfg.changed_ratio
        self.add_ratio = cfg.add_ratio

    def gen_adj(self, goal, all_edges, adj, clean_features):
        """生成子图的邻接矩阵和特征"""
        subset, edge_index, _, _ = k_hop_subgraph(
            goal, self.cfg.layer_numbers, all_edges, relabel_nodes=False, num_nodes=None
        )
        submapping, reverse_mapping, map_edge_index, map_edge_weight = subadj_map(subset, edge_index, adj)
        sub_features = subfeaturs(clean_features, reverse_mapping)

        edges_old_dict = dict()
        for i in range(len(map_edge_index[0])):
            edges_old_dict[str(map_edge_index[0][i]) + ',' + str(map_edge_index[1][i])] = i

        sub_old = rumor_construct_adj_matrix_v2(map_edge_index, len(submapping), map_edge_weight)
        sub_old_nonzero = sub_old.nonzero()
        subgraph = matrixtodict(sub_old_nonzero)

        return submapping, sub_features, map_edge_index, map_edge_weight, edges_old_dict, subgraph, sub_old


# ========== 主运行类 ==========
class NodeExplainer:
    def __init__(self, cfg: ExplainConfig):
        self.cfg = cfg
        self.setup_seed()
        self.load_dataset()
        self.load_models()
        self.prepare_graph()
        self.gen_data = GenData(cfg)

        self._init_data_split()

    def _init_data_split(self):
        """初始化训练/验证数据集划分"""
        all_nodes = list(range(self.adj_old.shape[0]))
        target_goal_list = random.sample(
            all_nodes,
            min(self.cfg.sample_num_nodes, len(all_nodes))
        )

        split_idx = int(self.cfg.train_ratio * len(target_goal_list))
        self.train_nodes = target_goal_list[:split_idx]
        self.val_nodes = target_goal_list[split_idx:]

        print(f"Data split: train={len(self.train_nodes)}, val={len(self.val_nodes)}")

    def get_lambda_save_dir(self) -> str:
        """获取 lambda 相关文件的保存目录"""
        if self.cfg.normalize_adj:
            norm_dir = "normalize"
        else:
            norm_dir = "no_normalize"

        print('norm_dir',norm_dir)

        save_dir = os.path.join(
            self.cfg.result_root,
            norm_dir,
            "ricci",
            self.modelname,
            self.cfg.method
        )
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def verify_train_node(self, node_idx: int) -> float:
        """
        验证节点是否适合训练
        返回 edge_percentage，值越小说明边重要性分布越不均匀，节点越适合训练
        """

        edge_mask = self.explain_node(node_idx, use_cache=True)
        if edge_mask is None:
            return 1.0  # 返回高值表示不适合
        print('edge_mask',edge_mask)
        percentage = edge_percentage(edge_mask)
        return percentage


    def run_train_mode(self):
        """筛选训练节点并保存"""
        print(f"[TRAIN MODE] Filtering {len(self.train_nodes)} nodes...")

        valid_nodes = []
        for node_idx in self.train_nodes:
            percentage = self.verify_train_node(node_idx)
            if percentage < self.cfg.percentage_threshold:
                valid_nodes.append(node_idx)
                print(f"Node {node_idx}: VALID (percentage={percentage:.4f})")
            else:
                print(f"Node {node_idx}: SKIP (percentage={percentage:.4f})")

        # 保存训练节点
        save_dir = self.get_lambda_save_dir()
        save_path = os.path.join(save_dir, "train_node.json")
        with open(save_path, 'w') as f:
            json.dump(valid_nodes, f)

        print(f"[TRAIN MODE] Saved {len(valid_nodes)} nodes to {save_path}")
        return valid_nodes

    def run_val_mode(self):
        """筛选验证节点并保存"""
        print(f"[VAL MODE] Filtering {len(self.val_nodes)} nodes...")

        valid_nodes = []
        for node_idx in self.val_nodes:
            percentage = self.verify_train_node(node_idx)
            if percentage < self.cfg.percentage_threshold:
                valid_nodes.append(node_idx)
                print(f"Node {node_idx}: VALID (percentage={percentage:.4f})")
            else:
                print(f"Node {node_idx}: SKIP (percentage={percentage:.4f})")

        # 保存验证节点
        save_dir = self.get_lambda_save_dir()
        save_path = os.path.join(save_dir, "val_node.json")
        with open(save_path, 'w') as f:
            json.dump(valid_nodes, f)

        print(f"[VAL MODE] Saved {len(valid_nodes)} nodes to {save_path}")
        return valid_nodes

    def run_select_mode(self):
        """使用 Optuna 优化 lambda 参数"""
        save_dir = self.get_lambda_save_dir()
        train_node_path = os.path.join(save_dir, "train_node.json")

        # 加载训练节点
        if not os.path.exists(train_node_path):
            print(f"[SELECT MODE] train_node.json not found, running TRAIN mode first...")
            self.run_train_mode()

        with open(train_node_path, 'r') as f:
            target_nodes = json.load(f)

        # 限制节点数量
        if len(target_nodes) > self.cfg.max_train_nodes:
            target_nodes = random.sample(target_nodes, self.cfg.max_train_nodes)

        print(f"[SELECT MODE] Optimizing lambda with {len(target_nodes)} nodes...")

        good_lambdas = []

        def objective(trial):
            lam = trial.suggest_float("lambda", self.cfg.lambda_min, self.cfg.lambda_max)

            # 计算所有节点的 prob_robust 差值
            probs = []
            for node_idx in target_nodes:
                prob = self.evaluate_node_with_lambda(node_idx, lam)
                if prob is not None:
                    probs.append(prob)


            if len(probs) == 0:
                return 0.0

            avg_prob = np.mean(probs)

            # 如果平均值 < 0，说明 Ricci 增强有帮助
            if avg_prob < 0:
                good_lambdas.append(lam)
                print(f"Lambda {lam:.4f}: avg_prob={avg_prob:.4f} (GOOD)")

                # 保存 good_lambdas
                with open(os.path.join(save_dir, "good_lambdas.json"), 'w') as f:
                    json.dump(good_lambdas, f)

            return avg_prob

        # 运行 Optuna 优化
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.cfg.n_trials)

        print(f"[SELECT MODE] Best lambda: {study.best_params['lambda']}")
        print(f"[SELECT MODE] Good lambdas: {good_lambdas}")

        # 保存最佳参数
        result = {
            "best_lambda": study.best_params['lambda'],
            "best_value": study.best_value,
            "good_lambdas": good_lambdas
        }
        with open(os.path.join(save_dir, "optuna_result.json"), 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def val_robustness(self, edges_index, edges_weight, select_edges, num_add, num_remove,
                       subadj, num_val, all_edges, submapping, node_idx, sub_features, output_goal):
        """评估选定边的鲁棒性"""
        result_dict = dict()
        predict_old_label = np.argmax(softmax(output_goal[submapping[node_idx]].detach().numpy()))

        for count in range(num_val):
            edges_old_dict = dict()
            for i in range(len(edges_index[0])):
                edge_str = str(edges_index[0][i].item()) + ',' + str(edges_index[1][i].item())
                edges_old_dict[edge_str] = i

            edges_old_dict_reverse = {v: [int(k.split(',')[0]), int(k.split(',')[1])]
                                      for k, v in edges_old_dict.items()}

            edges_weight_new = copy.deepcopy(edges_weight).tolist()
            edges_index_new = copy.deepcopy(edges_index).tolist()

            tmp_changed_edgelist = []
            change_num = 0

            # 随机移除边（不移除选中的重要边）
            while change_num < num_remove:
                random_edge_list = random.sample(list(range(len(edges_old_dict))), 1)
                for random_edge in random_edge_list:
                    remove_node_list = edges_old_dict_reverse[random_edge]
                    if ([remove_node_list[0], remove_node_list[1]] not in select_edges and
                            [remove_node_list[1], remove_node_list[0]] not in select_edges and
                            [remove_node_list[0], remove_node_list[1]] not in tmp_changed_edgelist and
                            [remove_node_list[1], remove_node_list[0]] not in tmp_changed_edgelist):

                        tmp_changed_edgelist.append([remove_node_list[0], remove_node_list[1]])
                        # 将边权重设为0（移除边）
                        if remove_node_list[0] != remove_node_list[1]:
                            idx1 = edges_old_dict[f"{remove_node_list[0]},{remove_node_list[1]}"]
                            idx2 = edges_old_dict[f"{remove_node_list[1]},{remove_node_list[0]}"]
                            edges_weight_new[idx1] = 0
                            edges_weight_new[idx2] = 0
                        else:
                            idx1 = edges_old_dict[f"{remove_node_list[0]},{remove_node_list[1]}"]
                            edges_weight_new[idx1] = 0
                        change_num += 1

            # 随机添加边
            add_num = 0
            while add_num < num_add:
                node1 = random.randint(0, subadj.shape[0] - 1)
                node2 = random.randint(0, subadj.shape[0] - 1)
                if (subadj[node1, node2] == 0 and
                        [node1, node2] not in tmp_changed_edgelist and
                        [node2, node1] not in tmp_changed_edgelist):

                    tmp_changed_edgelist.append([node1, node2])
                    # 添加新边
                    if node1 != node2:
                        edges_index_new[0].append(node1)
                        edges_index_new[1].append(node2)
                        edges_weight_new.append(1.0)
                        edges_index_new[0].append(node2)
                        edges_index_new[1].append(node1)
                        edges_weight_new.append(1.0)
                    else:
                        edges_index_new[0].append(node1)
                        edges_index_new[1].append(node2)
                        edges_weight_new.append(1.0)
                    add_num += 1

            # 评估扰动后的预测
            edges_index_tensor = torch.tensor(edges_index_new, dtype=torch.long)
            edges_weight_tensor = torch.tensor(edges_weight_new, dtype=torch.float32)

            output_new = self.model_gnn.forward(sub_features, edges_index_tensor, edges_weight_tensor)
            predict_new_label = np.argmax(softmax(output_new[submapping[node_idx]].detach().numpy()))

            # 记录是否预测改变
            if predict_new_label != predict_old_label:
                result_dict[count] = 1
            else:
                result_dict[count] = 0

        return result_dict

    def ave(self, my_dict):
        """计算字典值的平均"""
        non_zero_values = [v for v in my_dict.values() if v != float('inf')]
        if non_zero_values:
            return sum(non_zero_values) / len(non_zero_values), len(non_zero_values)
        return 0, 0

    def evaluate_node_with_lambda(self, node_idx: int, lam: float) -> Optional[float]:
        """
        评估节点在给定 lambda 下的鲁棒性差值
        返回: prob_robust(ricci) - prob_robust(base)
        如果 ricci 增强后更鲁棒，返回负值
        """
        # 获取边掩码
        edge_mask = self.explain_node(node_idx, use_cache=True)
        if edge_mask is None:
            return None

        # 获取子图
        submapping, sub_features, map_edge_index, map_edge_weight, edges_dict, subgraph, sub_adj = \
            self.gen_data.gen_adj(node_idx, torch.tensor(self.edges_old), self.adj_old, self.x.detach().numpy())

        # 转换为 tensor
        if isinstance(sub_features, np.ndarray):
            sub_features = torch.tensor(sub_features, dtype=torch.float32)
        if isinstance(map_edge_index, list):
            map_edge_index_tensor = torch.tensor(map_edge_index, dtype=torch.long)
        else:
            map_edge_index_tensor = map_edge_index
        if isinstance(map_edge_weight, list):
            map_edge_weight_tensor = torch.tensor(map_edge_weight, dtype=torch.float32)
        else:
            map_edge_weight_tensor = map_edge_weight

        # 计算 Ricci 曲率
        ricci_result = self.compute_ricci_curvature(map_edge_index_tensor, sub_adj)

        # 模型前向传播
        output_old = self.model_gnn.forward(sub_features, map_edge_index_tensor, map_edge_weight_tensor)
        predict_old_label = np.argmax(softmax(output_old[submapping[node_idx]].detach().numpy()))

        # 获取路径和目标边
        paths = dfs2(submapping[node_idx], submapping[node_idx], subgraph, self.cfg.layer_numbers + 1, [], [])
        paths = reverse_paths(paths)

        target_edgelist = []
        for path in paths:
            if [path[0], path[1]] not in target_edgelist and [path[1], path[0]] not in target_edgelist:
                target_edgelist.append([path[0], path[1]])
            if [path[2], path[1]] not in target_edgelist and [path[1], path[2]] not in target_edgelist:
                target_edgelist.append([path[1], path[2]])
        target_edgelist = clear(target_edgelist)

        # 计算选择的边数量
        select_edge_important = math.ceil(0.1 * len(target_edgelist))
        if select_edge_important < 1:
            return None

        # 准备评估参数
        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))
        num_val = 100  # 验证次数

        if num_remove_val <= 0 or num_add_val <= 0:
            return None

        # 构建边索引和权重
        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in target_edgelist:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_adj[edge[0], edge[1]])
                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(sub_adj[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_adj[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight, dtype=torch.float32)

        # ========== 基础选择（不使用 Ricci）==========
        total_result_base = dict()
        for i in range(len(map_edge_index[0])):
            edge_str = str(map_edge_index[0][i]) + ',' + str(map_edge_index[1][i])
            total_result_base[edge_str] = edge_mask[i].item()

        sort_diff_base = sorted(total_result_base.items(), key=lambda x: x[1], reverse=True)

        select_gnn_base_edgelist = []
        select_idx = 0
        while len(select_gnn_base_edgelist) < select_edge_important and select_idx < len(sort_diff_base):
            edge_str = sort_diff_base[select_idx][0]
            nodes = [int(n) for n in edge_str.split(',')]
            if nodes not in select_gnn_base_edgelist and [nodes[1], nodes[0]] not in select_gnn_base_edgelist:
                select_gnn_base_edgelist.append(nodes)
            select_idx += 1

        # ========== Ricci 增强选择 ==========
        total_result_ricci = dict()
        for i in range(len(map_edge_index[0])):
            edge_str = str(map_edge_index[0][i]) + ',' + str(map_edge_index[1][i])
            ricci_val = ricci_result.get(edge_str, 0)
            total_result_ricci[edge_str] = edge_mask[i].item() + lam * ricci_val

        sort_diff_ricci = sorted(total_result_ricci.items(), key=lambda x: x[1], reverse=True)

        select_gnn_edgelist = []
        select_idx = 0
        while len(select_gnn_edgelist) < select_edge_important and select_idx < len(sort_diff_ricci):
            edge_str = sort_diff_ricci[select_idx][0]
            nodes = [int(n) for n in edge_str.split(',')]
            if nodes not in select_gnn_edgelist and [nodes[1], nodes[0]] not in select_gnn_edgelist:
                select_gnn_edgelist.append(nodes)
            select_idx += 1

        # ========== 检查选择是否不同 ==========
        select_flag = True
        for edge in select_gnn_edgelist:
            if edge not in select_gnn_base_edgelist and [edge[1], edge[0]] not in select_gnn_base_edgelist:
                select_flag = False
                break

        if select_flag:
            # 选择相同，返回 0
            return 0

        # ========== 评估鲁棒性 ==========
        # Ricci 增强后的鲁棒性
        result_val_prob_ricci = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
            num_add_val, num_remove_val, sub_adj, num_val, len(target_edgelist),
            submapping, node_idx, sub_features, output_old
        )
        prob_robust, _ = self.ave(result_val_prob_ricci)

        # 基础选择的鲁棒性
        result_val_prob_base = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
            num_add_val, num_remove_val, sub_adj, num_val, len(target_edgelist),
            submapping, node_idx, sub_features, output_old
        )
        prob_robust_base, _ = self.ave(result_val_prob_base)

        # 返回差值：如果 Ricci 增强更鲁棒，prob_robust 更小，差值为负
        return prob_robust - prob_robust_base



    def run(self):
        """根据配置的模式运行"""
        mode = RunMode(self.cfg.run_mode)

        if mode == RunMode.TRAIN:
            return self.run_train_mode()
        elif mode == RunMode.VAL:
            return self.run_val_mode()
        elif mode == RunMode.SELECT:
            return self.run_select_mode()
        else:
            raise ValueError(f"Unknown run mode: {self.cfg.run_mode}")





    def setup_seed(self):
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def load_dataset(self):
        """加载数据集"""
        dataset_paths = {
            'Cora': ('PlanetoidCora', 'Cora'),
            'Citeseer': ('PlanetoidCiteseer', 'Citeseer'),
            'PubMed': ('pubmed', 'PubMed'),
        }

        if self.cfg.dataset not in dataset_paths:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset}")

        folder, name = dataset_paths[self.cfg.dataset]
        dataset_path = os.path.join(self.cfg.data_root, folder)
        self.dataset = Planetoid(dataset_path, name)
        self.data = self.dataset[0]

        self.x = self.data.x
        self.edge_index = self.data.edge_index
        self.labels = self.data.y
        self.num_classes = self.labels.max().item() + 1
        self.modelname = self.cfg.dataset.lower()

        print(f"Loaded dataset: {self.cfg.dataset}")
        print(f"  Nodes: {self.x.shape[0]}, Features: {self.x.shape[1]}, Classes: {self.num_classes}")

    def load_models(self):
        """加载预训练模型并转换权重"""
        # 加载原始 GCN 模型
        if self.cfg.normalize_adj:
            model_path = f'./checkpoints/node_{self.modelname}_norm.pt'

        else:
            model_path = f'./checkpoints/node_{self.modelname}_nonorm.pt'

        print('model_path', model_path)



        self.model = GCN(
            nfeat=self.x.shape[1],
            nhid=self.cfg.hidden_dim,
            nclass=self.num_classes,
            dropout=self.cfg.dropout
        )
        self.model.eval()

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state'])
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")

        # 创建用于解释的 GCN_explainer 模型
        self.model_gnn = GCN_explainer(
            nfeat=self.x.shape[1],
            nhid=self.cfg.hidden_dim,
            nclass=self.num_classes,
            dropout=self.cfg.dropout
        )
        self.model_gnn.eval()

        # 权重转换
        model_dict = self.model_gnn.state_dict()
        model_dict['conv1.lin.weight'] = self.model.state_dict()['gc1.weight'].t()
        model_dict['conv2.lin.weight'] = self.model.state_dict()['gc2.weight'].t()
        # model_dict['conv1.weight'] = self.model.state_dict()['gc1.weight']
        # model_dict['conv2.weight'] = self.model.state_dict()['gc2.weight']
        self.model_gnn.load_state_dict(model_dict)

        # 提取权重矩阵
        self.W1 = self.model_gnn.state_dict()['conv1.lin.weight'].t()
        self.W2 = self.model_gnn.state_dict()['conv2.lin.weight'].t()

    def prepare_graph(self):
        """准备图数据（添加自环）"""
        self.edges_old = copy.deepcopy(self.edge_index.tolist())
        for i in range(self.x.shape[0]):
            self.edges_old[0].append(i)
            self.edges_old[1].append(i)

        if self.cfg.normalize_adj:
            self.adj_old = rumor_construct_adj_matrix(self.edges_old, self.x.shape[0])
        else:
            edge_weight = [1.0] * len(self.edges_old[0])
            self.adj_old = rumor_construct_adj_matrix_v2(self.edges_old, self.x.shape[0], edge_weight)

        # self.adj_old = rumor_construct_adj_matrix(self.edges_old, self.x.shape[0])
        print(f"Adjacency matrix shape: {self.adj_old.shape}")

    def get_save_dir(self, subdir: str = None) -> str:
        """获取保存目录"""
        if subdir is None:
            if self.cfg.normalize_adj:
                subdir = "edge_masks_normalize"
            else:
                subdir = "edge_masks"
        save_dir = os.path.join(self.cfg.result_root, subdir, self.modelname, self.cfg.method)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def compute_ricci_curvature(self, map_edge_index, sub_adj) -> dict:
        """计算 Ollivier-Ricci 曲率"""
        if isinstance(map_edge_index, list):
            edge_index_tensor = torch.tensor(map_edge_index, dtype=torch.long)
        elif isinstance(map_edge_index, torch.Tensor):
            edge_index_tensor = map_edge_index
        else:
            edge_index_tensor = torch.tensor(list(map_edge_index), dtype=torch.long)

        old_data = Data(edge_index=edge_index_tensor, num_nodes = sub_adj.shape[0])
        G_old = to_networkx(old_data, to_undirected=True)

        orc = OllivierRicci(G_old, alpha=self.cfg.ricci_alpha)
        orc.compute_ricci_curvature()

        ricci_result = dict()
        for node_1, node_2 in orc.G.edges:
            curv = normalize_ricci(orc.G[node_1][node_2]['ricciCurvature'])
            ricci_result[f"{node_1},{node_2}"] = curv
            ricci_result[f"{node_2},{node_1}"] = curv

        # 自环使用中位数
        median_number = sum(ricci_result.values()) / len(ricci_result) if ricci_result else 0
        for i in range(sub_adj.shape[0]):
            ricci_result[f"{i},{i}"] = median_number

        return ricci_result

    def explain_node(self, node_idx: int, use_cache: bool = True) -> Optional[torch.Tensor]:
        """对单个节点进行解释"""
        # 生成子图
        submapping, sub_features, map_edge_index, map_edge_weight, edges_dict, subgraph, sub_adj = \
            self.gen_data.gen_adj(node_idx, torch.tensor(self.edges_old), self.adj_old, self.x.detach().numpy())

        if isinstance(sub_features, np.ndarray):
            sub_features = torch.tensor(sub_features, dtype=torch.float32)
        if isinstance(map_edge_index, list):
            map_edge_index = torch.tensor(map_edge_index, dtype=torch.long)
        if isinstance(map_edge_weight, list):
            map_edge_weight = torch.tensor(map_edge_weight, dtype=torch.float32)

        # 模型前向传播
        output = self.model_gnn.forward(sub_features, map_edge_index, map_edge_weight)
        predict_label = np.argmax(softmax(output[submapping[node_idx]].detach().numpy()))

        # 获取路径
        paths = dfs2(submapping[node_idx], submapping[node_idx], subgraph, self.cfg.layer_numbers + 1, [], [])
        paths = reverse_paths(paths)

        # 获取目标边列表
        target_edgelist = []
        for path in paths:
            if [path[0], path[1]] not in target_edgelist and [path[1], path[0]] not in target_edgelist:
                target_edgelist.append([path[0], path[1]])
            if [path[2], path[1]] not in target_edgelist and [path[1], path[2]] not in target_edgelist:
                target_edgelist.append([path[1], path[2]])
        target_edgelist = clear(target_edgelist)

        # 检查是否满足条件
        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))

        if len(target_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Node {node_idx}: Skipped (edges={len(target_edgelist)})")
            return None

        # 缓存路径
        save_dir = self.get_save_dir()
        save_path = os.path.join(save_dir, f"{self.cfg.method}_{node_idx}.npy")

        if use_cache and os.path.exists(save_path):
            edge_mask_array = np.load(save_path)
            edge_mask = torch.tensor(edge_mask_array, dtype=torch.float32)
            print(f"Node {node_idx}: Loaded from cache")
        else:
            # 创建解释器并计算
            node_idx_mapped = submapping[node_idx]

            if self.cfg.method == "deeplift":
                # print('deeplift',deeplift)
                explainer = DeepLIFT(self.model_gnn, explain_graph=False)
                sparsity = 1
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        num_classes=self.num_classes, node_idx=node_idx_mapped,
                                        edge_weight=map_edge_weight)
                edge_mask = masks[predict_label]

            elif self.cfg.method == "flowx":
                explainer = FlowMask(self.model_gnn, explain_graph=False)
                sparsity = 0
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        num_classes=self.num_classes, node_idx=node_idx_mapped,
                                        edge_weight=map_edge_weight)
                edge_mask = masks[predict_label]

            elif self.cfg.method == "gnnlrp":
                explainer = GNN_LRP(self.model_gnn, explain_graph=False)
                sparsity = 0
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        node_idx=node_idx_mapped, edge_weight=map_edge_weight,
                                        given_class=predict_label)
                edge_mask = masks[0]
                edge_mask = custom_log_transform(edge_mask)
            else:
                # 其他方法使用 explainer_unified
                explainer = ExplainerFactory.create(self.cfg.method, self.model_gnn)
                edge_mask = explainer.explain(
                    sub_features, map_edge_index,
                    node_idx=node_idx_mapped,
                    edge_weight=map_edge_weight,
                    num_classes=self.num_classes,
                    target_class=predict_label
                )



            # 后处理（如果是 GNN-LRP 则进行 log 变换）
            if self.cfg.method == "gnnlrp":
                edge_mask = custom_log_transform(edge_mask)

            # 保存
            np.save(save_path, edge_mask.detach().cpu().numpy())
            print(f"Node {node_idx}: Computed and saved")

        return edge_mask

    def explain_with_ricci(self, node_idx: int, lam: float = None) -> Optional[float]:
        """结合 Ricci 曲率的解释"""
        if lam is None:
            lam = self.cfg.ricci_lambda

        # 生成子图
        submapping, sub_features, map_edge_index, map_edge_weight, edges_dict, subgraph, sub_adj = \
            self.gen_data.gen_adj(node_idx, torch.tensor(self.edges_old), self.adj_old, self.x.detach().numpy())

        # 计算 Ricci 曲率
        ricci_result = self.compute_ricci_curvature(map_edge_index, sub_adj)

        print('ricci_result',ricci_result)

        # 获取解释掩码
        edge_mask = self.explain_node(node_idx)
        if edge_mask is None:
            return None

        # 结合 Ricci 曲率评估
        test_percentage = edge_percentage(edge_mask)
        print(f"Node {node_idx}: Edge percentage = {test_percentage:.4f}")

        return test_percentage

    def run_batch(self, node_list: List[int] = None):
        """批量处理节点"""
        if node_list is None:
            # 随机采样节点
            node_list = random.sample(
                list(range(self.adj_old.shape[0])),
                min(self.cfg.sample_num_nodes, self.adj_old.shape[0])
            )

        print('node_list',node_list)

        results = []
        for node_idx in node_list:
            result = self.explain_with_ricci(node_idx)
            results.append((node_idx, result))
        return results


# ========== 命令行解析 ==========
def parse_args() -> ExplainConfig:
    p = argparse.ArgumentParser(description="Unified Node Explanation with Ricci Curvature")

    # 数据集
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed"])
    p.add_argument("--data_root", type=str, default="./data")

    # 模型
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--layer_numbers", type=int, default=2)

    # 解释方法
    p.add_argument("--method", type=str, default="gnnlrp",
                   choices=["deeplift", "flowx", "gnnexplainer", "gnnlrp", "pgexplainer",'convex'])

    # Ricci
    p.add_argument("--ricci_alpha", type=float, default=0.0)
    p.add_argument("--ricci_lambda", type=float, default=0.0)

    # 采样
    p.add_argument("--sample_num_nodes", type=int, default=500)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--percentage_threshold", type=float, default=0.8)

    # 扰动
    p.add_argument("--changed_ratio", type=float, default=0.5)
    p.add_argument("--add_ratio", type=float, default=0.5)
    p.add_argument("--num_remove_val_ratio", type=float, default=0.1)
    p.add_argument("--num_add_val_ratio", type=float, default=0.1)

    # 保存
    p.add_argument("--result_root", type=str, default="./result")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--normalize_adj",  default=False)  # 新增

    p.add_argument("--run_mode", type=str, default="select",
                   choices=["train", "val", "select"],
                   help="运行模式: train=筛选训练节点, val=筛选验证节点, select=优化lambda")

    # Optuna 配置
    p.add_argument("--n_trials", type=int, default=50, help="Optuna 试验次数")
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.1)
    p.add_argument("--max_train_nodes", type=int, default=250)

    args = p.parse_args()

    return ExplainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        layer_numbers=args.layer_numbers,
        method=args.method,
        ricci_alpha=args.ricci_alpha,
        ricci_lambda=args.ricci_lambda,
        sample_num_nodes=args.sample_num_nodes,
        train_ratio=args.train_ratio,
        percentage_threshold=args.percentage_threshold,
        changed_ratio=args.changed_ratio,
        add_ratio=args.add_ratio,
        num_remove_val_ratio=args.num_remove_val_ratio,
        num_add_val_ratio=args.num_add_val_ratio,
        result_root=args.result_root,
        seed=args.seed,
        device=args.device,
        run_mode=args.run_mode,
        n_trials=args.n_trials,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        max_train_nodes=args.max_train_nodes,
        normalize_adj=args.normalize_adj
    )


# ========== 主函数 ==========
def main():
    cfg = parse_args()
    print(f"Config: {cfg}")

    explainer = NodeExplainer(cfg)

    # 运行批量解释
    results = explainer.run()

    # 保存结果统计



if __name__ == "__main__":
    main()