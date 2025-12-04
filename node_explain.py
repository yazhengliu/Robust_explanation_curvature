import os
import sys
import copy
import math
import json
import random
import argparse
import optuna
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from enum import Enum

from utils.models import GCN_explainer,GCN_explainer_pyg
from utils.node_utils import (
    rumor_construct_adj_matrix, matrixtodict, clear, softmax,
    k_hop_subgraph, subadj_map, subfeaturs, rumor_construct_adj_matrix_v2,
    dfs2, reverse_paths, edge_percentage, custom_log_transform,normalize_ricci, from_edges_to_evaulate,
KL_divergence, test_path_contribution_edge, map_target, main_con_edge
)
from utils.Network import Network
from torch_geometric.utils.loop import remove_self_loops

from scipy.sparse import csr_matrix
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

    curvature_type: str = "ricci"

    sparsity: float = 0.1

    # Ricci 曲率配置
    ricci_alpha: float = 0.0
    ricci_lambda: float = 0.0

    # 节点采样配置
    sample_num_nodes: int = 500
    train_ratio: float = 0.5
    percentage_threshold: float = 0.8
    resistance_epsilon:float=0.01
    resistance_method: str='kts'

    n_trials: int = 50
    lambda_min: float = 0.0
    lambda_max: float = 0.1
    max_train_nodes: int = 250


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

    def gen_parameters_v2(self, features, edges_tensor, edge_weight, model):
        """计算 convex 方法所需的 relu 参数"""
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_tensor, edge_weight)
        nonlinear_start_layer1 = torch.zeros_like(nonlinear_end_layer1)
        nonlinear_relu_start_layer1 = torch.zeros_like(nonlinear_relu_end_layer1)

        relu_delta = torch.where(
            (nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
            (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (nonlinear_end_layer1 - nonlinear_start_layer1),
            torch.zeros_like(nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)
        )
        relu_end = torch.where(
            nonlinear_end_layer1 != 0,
            nonlinear_relu_end_layer1 / nonlinear_end_layer1,
            torch.zeros_like(nonlinear_end_layer1)
        )
        relu_start = torch.where(
            nonlinear_start_layer1 != 0,
            nonlinear_relu_start_layer1 / nonlinear_start_layer1,
            torch.zeros_like(nonlinear_start_layer1)
        )
        return relu_delta, relu_end, relu_start


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

        # print('norm_dir', norm_dir)

        save_dir = os.path.join(
            self.cfg.result_root,
            norm_dir,
            self.cfg.curvature_type,  # 使用 curvature_type 替代硬编码的 "ricci"
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

        save_dir = self.get_lambda_save_dir()

        val_node_path = os.path.join(save_dir, "val_node.json")
        if not os.path.exists(val_node_path):
            print(f"[VAL MODE] val_node.json not found, generating...")
            # 筛选验证节点
            valid_nodes = []
            for node_idx in self.val_nodes:
                percentage = self.verify_train_node(node_idx)
                if percentage < self.cfg.percentage_threshold:
                    valid_nodes.append(node_idx)
                    print(f"Node {node_idx}: VALID (percentage={percentage:.4f})")

            with open(val_node_path, 'w') as f:
                json.dump(valid_nodes, f)
        else:
            with open(val_node_path, 'r') as f:
                valid_nodes = json.load(f)

        print(f"[VAL MODE] Loaded {len(valid_nodes)} validation nodes")

        good_lambdas_path = os.path.join(save_dir, "good_lambdas.json")
        if not os.path.exists(good_lambdas_path):
            print(f"[VAL MODE] good_lambdas.json not found, please run SELECT mode first")
            return None

        with open(good_lambdas_path, 'r') as f:
            good_lambdas_list = json.load(f)

        print(f"[VAL MODE] Loaded {len(good_lambdas_list)} good lambdas: {good_lambdas_list}")

        for node_idx in valid_nodes:
            print(f"[VAL MODE] Evaluating node {node_idx}...")
            self.evaluate_val_node(node_idx, good_lambdas_list)



        print(f"[VAL MODE] Completed evaluation for {len(valid_nodes)} nodes")
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

        result = {
            "best_lambda": study.best_params['lambda'],
            "best_value": study.best_value,
            "good_lambdas": good_lambdas
        }

        return result

    def val_robustness(self, edges_index, edges_weight, select_edges, num_add, num_remove,
                       subadj, num_val, all_edges, submapping, node_idx, sub_features, output_goal):
        """评估选定边的鲁棒性（与 deeplift_ricci_train.py 完全一致）"""
        result_dict = dict()
        result_dict_KL = dict()

        predict_old_label = np.argmax(softmax(output_goal[submapping[node_idx]].detach().numpy()))

        for count in range(num_val):
            tmp_changed_edgelist = []
            edges_old_dict = dict()
            for i in range(len(edges_index[0])):
                edges_old_dict[str(edges_index[0][i].item()) + ',' + str(edges_index[1][i].item())] = i

            edges_old_dict_reverse = dict()
            for key, value in edges_old_dict.items():
                node_list = key.split(',')
                edges_old_dict_reverse[value] = [int(node_list[0]), int(node_list[1])]

            # 限制添加边的数量
            add_bound = subadj.shape[0] * (subadj.shape[0] + 1) / 2 - all_edges
            num_add_actual = min(num_add, add_bound)

            edges_weight_new = copy.deepcopy(edges_weight).tolist()
            edges_index_new = copy.deepcopy(edges_index).tolist()

            # 随机移除边（不移除选中的重要边）
            change_num = 0
            while change_num < num_remove:
                random_edge_list = random.sample(list(range(len(edges_old_dict))), 1)

                for random_edge in random_edge_list:
                    remove_node_list = edges_old_dict_reverse[random_edge]

                    if ([remove_node_list[0], remove_node_list[1]] not in select_edges and
                            [remove_node_list[1], remove_node_list[0]] not in select_edges and
                            [remove_node_list[0], remove_node_list[1]] not in tmp_changed_edgelist and
                            [remove_node_list[1], remove_node_list[0]] not in tmp_changed_edgelist):
                        index_1 = str(remove_node_list[0]) + ',' + str(remove_node_list[1])
                        edges_weight_new[edges_old_dict[index_1]] = 0

                        index_2 = str(remove_node_list[1]) + ',' + str(remove_node_list[0])
                        edges_weight_new[edges_old_dict[index_2]] = 0

                        tmp_changed_edgelist.append([remove_node_list[0], remove_node_list[1]])
                        change_num += 1

            # 随机添加边（不添加已选择的重要边）
            change_num = 0
            while change_num < num_add_actual:
                node_x = random.sample(list(range(subadj.shape[0])), 1)[0]
                node_y = random.sample(list(range(subadj.shape[0])), 1)[0]

                str_edge = str(node_x) + ',' + str(node_y)
                if str_edge not in edges_old_dict.keys():
                    if ([node_x, node_y] not in tmp_changed_edgelist and
                            [node_y, node_x] not in tmp_changed_edgelist and
                            [node_x, node_y] not in select_edges and
                            [node_y, node_x] not in select_edges):

                        tmp_changed_edgelist.append([node_x, node_y])
                        if node_x != node_y:
                            edges_index_new[1].append(node_x)
                            edges_index_new[0].append(node_y)
                            edges_index_new[0].append(node_x)
                            edges_index_new[1].append(node_y)
                            edges_weight_new.append(1)
                            edges_weight_new.append(1)
                        else:
                            edges_index_new[0].append(node_x)
                            edges_index_new[1].append(node_y)
                            edges_weight_new.append(1)

                        change_num += 1

            # 评估扰动后的预测
            tmp_output = self.model_gnn.forward(
                sub_features,
                torch.tensor(edges_index_new),
                edge_weight=torch.tensor(edges_weight_new, dtype=torch.float32)
            )

            # 计算概率差值（与原始代码一致）
            result_dict[count] = float(abs(
                softmax(tmp_output[submapping[node_idx]].detach().numpy())[predict_old_label] -
                softmax(output_goal[submapping[node_idx]].detach().numpy())[predict_old_label]
            ))

            # 计算 KL 散度
            result_dict_KL[count] = KL_divergence(
                softmax(output_goal[submapping[node_idx]].detach().numpy()),
                softmax(tmp_output[submapping[node_idx]].detach().numpy())
            )

        return result_dict, result_dict_KL

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
        curvature_result = self.compute_curvature(map_edge_index, map_edge_weight, sub_adj)

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
        select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))
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
        if self.cfg.method == "convex":
            # convex 方法: edge_mask 对应 target_edgelist
            edge_index_for_mask = target_edgelist
            total_result_base = dict()
            for i, edge in enumerate(target_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                if i < len(edge_mask):
                    total_result_base[edge_str] = edge_mask[i].item()
        else:
            # 其他方法: edge_mask 对应 map_edge_index
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
        if self.cfg.method == "convex":
            total_result_curvature = dict()
            for i, edge in enumerate(target_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                curvature_val = curvature_result[edge_str]
                if i < len(edge_mask):
                    total_result_curvature[edge_str] = self.compute_combined_score(
                        edge_mask[i].item(), curvature_val, lam
                    )
        else:
            total_result_curvature = dict()
            for i in range(len(map_edge_index[0])):
                edge_str = str(map_edge_index[0][i]) + ',' + str(map_edge_index[1][i])
                curvature_val = curvature_result[edge_str]
                total_result_curvature[edge_str] = self.compute_combined_score(
                    edge_mask[i].item(), curvature_val, lam
                )

        sort_diff_ricci = sorted(total_result_curvature.items(), key=lambda x: x[1], reverse=True)

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
        result_val_prob_ricci, result_val_kl_ricci = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
            num_add_val, num_remove_val, sub_adj, num_val, len(target_edgelist),
            submapping, node_idx, sub_features, output_old
        )
        prob_robust, _ = self.ave(result_val_prob_ricci)

        # 基础选择的鲁棒性
        result_val_prob_base, result_val_kl_base= self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
            num_add_val, num_remove_val, sub_adj, num_val, len(target_edgelist),
            submapping, node_idx, sub_features, output_old
        )
        prob_robust_base, _ = self.ave(result_val_prob_base)

        # 返回差值：如果 Ricci 增强更鲁棒，prob_robust 更小，差值为负
        return prob_robust - prob_robust_base

    def evaluate_val_node(self, node_idx: int, good_lambdas_list: list):

        # 获取子图
        submapping, sub_features, map_edge_index, map_edge_weight, edges_dict, subgraph, sub_adj = \
            self.gen_data.gen_adj(node_idx, torch.tensor(self.edges_old), self.adj_old, self.x.detach().numpy())

        # 转换为 tensor
        if isinstance(sub_features, np.ndarray):
            sub_features = torch.tensor(sub_features, dtype=torch.float32)
        if isinstance(map_edge_index, list):
            map_edge_index = torch.tensor(map_edge_index, dtype=torch.long)
        if isinstance(map_edge_weight, list):
            map_edge_weight = torch.tensor(map_edge_weight, dtype=torch.float32)

        # 模型前向传播
        output_old = self.model_gnn.forward(sub_features, map_edge_index, map_edge_weight)
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

        # 检查边数量条件
        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))

        if len(target_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Node {node_idx}: Skipped (edges={len(target_edgelist)})")
            return

        # 构建目标边的索引和权重
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

        # 计算 Ricci 曲率
        curvature_result = self.compute_curvature(map_edge_index, map_edge_weight, sub_adj)

        # 获取边掩码
        edge_mask = self.explain_node(node_idx, use_cache=True)
        if edge_mask is None:
            return

        # 边选择数量
        select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))

        # 随机选择边的鲁棒性（作为基准）
        random_select_edges = random.sample(target_edgelist, min(select_edge_important, len(target_edgelist)))
        withoutcon_result_val_prob, withoutcon_result_val_kl = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, random_select_edges,
            num_add_val, num_remove_val, sub_adj, 100, len(target_edgelist),
            submapping, node_idx, sub_features, output_old
        )

        # 对每个 lambda 进行评估
        for lam in good_lambdas_list:
            result_important_dict = dict()
            result_important_base_dict = dict()
            save_flag = False

            # ========== 基础选择（不使用 Ricci）==========
            if self.cfg.method == "convex":
                total_result_base = dict()
                for i, edge in enumerate(target_edgelist):
                    edge_str = str(edge[0]) + ',' + str(edge[1])
                    if i < len(edge_mask):
                        total_result_base[edge_str] = edge_mask[i].item()
            else:
                total_result_base = dict()
                for i in range(len(map_edge_index[0])):
                    edge_str = str(map_edge_index[0][i].item()) + ',' + str(map_edge_index[1][i].item())
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

            # ========== based curvature ==========
            if self.cfg.method == "convex":
                total_result_curvature = dict()
                for i, edge in enumerate(target_edgelist):
                    edge_str = str(edge[0]) + ',' + str(edge[1])
                    curvature_val = curvature_result[edge_str]
                    if i < len(edge_mask):
                        total_result_curvature[edge_str] = self.compute_combined_score(
                            edge_mask[i].item(), curvature_val, lam
                        )
            else:
                total_result_curvature = dict()
                for i in range(len(map_edge_index[0])):
                    edge_str = str(map_edge_index[0][i].item()) + ',' + str(map_edge_index[1][i].item())
                    curvature_val = curvature_result[edge_str]
                    total_result_curvature[edge_str] = self.compute_combined_score(
                        edge_mask[i].item(), curvature_val, lam
                    )

            sort_diff_ricci = sorted(total_result_curvature.items(), key=lambda x: x[1], reverse=True)

            select_gnn_edgelist = []
            select_idx = 0
            while len(select_gnn_edgelist) < select_edge_important and select_idx < len(sort_diff_ricci):
                edge_str = sort_diff_ricci[select_idx][0]
                nodes = [int(n) for n in edge_str.split(',')]
                if nodes not in select_gnn_edgelist and [nodes[1], nodes[0]] not in select_gnn_edgelist:
                    select_gnn_edgelist.append(nodes)
                select_idx += 1

            # 检查选择是否不同
            select_flag = True
            for edge in select_gnn_edgelist:
                if edge not in select_gnn_base_edgelist and [edge[1], edge[0]] not in select_gnn_base_edgelist:
                    select_flag = False
                    break

            if not select_flag:
                save_flag = True

                # 评估 Ricci 增强的选择
                evaluate_edge_index, evaluate_edge_weight = from_edges_to_evaulate(
                    select_gnn_edgelist, torch.tensor([]), [[], []], dict(),
                    csr_matrix(sub_adj.shape), sub_adj
                )
                evaluate_edge_index = torch.tensor(evaluate_edge_index).to(torch.int64)
                evaluate_edge_weight = torch.tensor(evaluate_edge_weight, dtype=torch.float32)

                evaluate_edge_output = self.model_gnn.forward(sub_features, evaluate_edge_index,
                                                              edge_weight=evaluate_edge_weight)

                KL_edge_ricci = KL_divergence(
                    softmax(output_old[submapping[node_idx]].detach().numpy()),
                    softmax(evaluate_edge_output[submapping[node_idx]].detach().numpy())
                )

                result_val_prob, result_val_kl = self.val_robustness(
                    target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
                    num_add_val, num_remove_val, sub_adj, 100, len(target_edgelist),
                    submapping, node_idx, sub_features, output_old
                )

                idx_important = 0
                result_important_dict[f'{idx_important},select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])] for e
                                                                                           in select_gnn_edgelist]
                result_important_dict[f'{idx_important},select {self.cfg.method} edgeKL'] = KL_edge_ricci
                result_important_dict[f'{idx_important},robustness {self.cfg.method} prob'] = result_val_prob
                result_important_dict[f'{idx_important},robustness {self.cfg.method} kl'] = result_val_kl
                result_important_dict[
                    f'{idx_important},robustness {self.cfg.method} prob'] = withoutcon_result_val_prob
                result_important_dict[
                    f'{idx_important},robustness {self.cfg.method} kl'] = withoutcon_result_val_kl

                # 评估基础选择
                evaluate_edge_index_base, evaluate_edge_weight_base = from_edges_to_evaulate(
                    select_gnn_base_edgelist, torch.tensor([]), [[], []], dict(),
                    csr_matrix(sub_adj.shape), sub_adj
                )
                evaluate_edge_index_base = torch.tensor(evaluate_edge_index_base).to(torch.int64)
                evaluate_edge_weight_base = torch.tensor(evaluate_edge_weight_base, dtype=torch.float32)

                evaluate_edge_output_base = self.model_gnn.forward(sub_features, evaluate_edge_index_base,
                                                                   edge_weight=evaluate_edge_weight_base)

                KL_edge_base = KL_divergence(
                    softmax(output_old[submapping[node_idx]].detach().numpy()),
                    softmax(evaluate_edge_output_base[submapping[node_idx]].detach().numpy())
                )

                result_val_prob_base, result_val_kl_base = self.val_robustness(
                    target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
                    num_add_val, num_remove_val, sub_adj, 100, len(target_edgelist),
                    submapping, node_idx, sub_features, output_old
                )

                result_important_base_dict[f'{idx_important},select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])]
                                                                                                for e in
                                                                                                select_gnn_base_edgelist]
                result_important_base_dict[f'{idx_important},select {self.cfg.method} edgeKL'] = KL_edge_base
                result_important_base_dict[f'{idx_important},robustness {self.cfg.method} prob'] = result_val_prob_base
                result_important_base_dict[f'{idx_important},robustness {self.cfg.method} kl'] = result_val_kl_base
                result_important_base_dict[
                    f'{idx_important},robustness {self.cfg.method} prob'] = withoutcon_result_val_prob
                result_important_base_dict[
                    f'{idx_important},robustness {self.cfg.method} kl'] = withoutcon_result_val_kl

            # 保存结果
            if save_flag:
                norm_dir = "normalize" if self.cfg.normalize_adj else "no_normalize"


                # 保存 Ricci 增强结果
                ricci_save_dir = os.path.join(
                    self.cfg.result_root, norm_dir, "ricci",
                    self.modelname, self.cfg.method,
                    f"{self.cfg.num_add_val_ratio}_{self.cfg.num_remove_val_ratio}",
                     str(lam), "ricci"
                )
                os.makedirs(ricci_save_dir, exist_ok=True)

                with open(os.path.join(ricci_save_dir, f"{node_idx}.json"), 'w') as f:
                    json.dump(result_important_dict, f)

                # 保存基础结果
                base_save_dir = os.path.join(
                    self.cfg.result_root,  norm_dir, "ricci",
                    self.modelname, self.cfg.method,
                    f"{self.cfg.num_add_val_ratio}_{self.cfg.num_remove_val_ratio}",
                     str(lam), "base"
                )
                os.makedirs(base_save_dir, exist_ok=True)

                with open(os.path.join(base_save_dir, f"{node_idx}.json"), 'w') as f:
                    json.dump(result_important_base_dict, f)

                print(f"Node {node_idx}, Lambda {lam}: Results saved")


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

        if self.cfg.method in ["deeplift", "flowx", "gnnlrp",'convex']:
            # DIG 方法使用自定义 GCNConv
            self.model_gnn = GCN_explainer(
                nfeat=self.x.shape[1],
                nhid=self.cfg.hidden_dim,
                nclass=self.num_classes,
                dropout=self.cfg.dropout
            )
        else:
            # PyG 方法（gnnexplainer, pgexplainer, convex）使用标准 GCNConv
            self.model_gnn = GCN_explainer_pyg(
                nfeat=self.x.shape[1],
                nhid=self.cfg.hidden_dim,
                nclass=self.num_classes,
                dropout=self.cfg.dropout
            )

        self.model_gnn.eval()

        # 创建用于解释的 GCN_explainer 模型
        # self.model_gnn = GCN_explainer(
        #     nfeat=self.x.shape[1],
        #     nhid=self.cfg.hidden_dim,
        #     nclass=self.num_classes,
        #     dropout=self.cfg.dropout
        # )
        # self.model_gnn.eval()

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

    def compute_combined_score(self, edge_mask_value: float, curvature_value: float, lam: float) -> float:
        """计算边的组合分数（解释器分数 + lambda * 曲率）"""
        if self.cfg.curvature_type == "ricci":
            # Ricci 曲率越高越重要
            return edge_mask_value + lam * curvature_value
        elif self.cfg.curvature_type == "resistance":
            # Resistance 越低越重要，所以用减法
            return edge_mask_value - lam * curvature_value


    def compute_curvature(self, map_edge_index, map_edge_weight, sub_adj) -> dict:

        """根据配置计算 Ricci 曲率或 Effective Resistance"""

        if isinstance(map_edge_index, list):
            map_edge_index = torch.tensor(map_edge_index, dtype=torch.long)
        if isinstance(map_edge_weight, list):
            map_edge_weight = torch.tensor(map_edge_weight, dtype=torch.float32)

        if self.cfg.curvature_type == "ricci":
            return self.compute_ricci_curvature(map_edge_index, sub_adj)
        elif self.cfg.curvature_type == "resistance":
            return self.compute_resistance(map_edge_index, map_edge_weight, sub_adj)
        else:
            raise ValueError(f"Unknown curvature type: {self.cfg.curvature_type}")

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

    def compute_resistance(self, map_edge_index, map_edge_weight, sub_adj) -> dict:
        """计算 Effective Resistance"""
        # 移除自环
        if isinstance(map_edge_index, list):
            map_edge_index = torch.tensor(map_edge_index, dtype=torch.long)
        if isinstance(map_edge_weight, list):
            map_edge_weight = torch.tensor(map_edge_weight, dtype=torch.float32)

        map_edge_index_remove, map_edge_weight_remove = remove_self_loops(map_edge_index, map_edge_weight)

        # 创建 networkx 图
        old_data_remove = Data(edge_index=map_edge_index_remove, edge_weight=map_edge_weight_remove,
                               num_nodes=sub_adj.shape[0])
        G_old = to_networkx(old_data_remove, to_undirected=True)

        # 计算 effective resistance
        network = Network(None, None, G_old)
        E_list, Effective_R = network.effR(self.cfg.resistance_epsilon, self.cfg.resistance_method)

        # 构建结果字典
        Effective_result = dict()
        total_r = 0
        total_count = 0

        for i in range(len(E_list)):
            if E_list[i][0] != E_list[i][1]:
                edge_str_1 = str(E_list[i][0]) + ',' + str(E_list[i][1])
                edge_str_2 = str(E_list[i][1]) + ',' + str(E_list[i][0])
                Effective_result[edge_str_1] = Effective_R[i]
                Effective_result[edge_str_2] = Effective_R[i]
                total_r += Effective_R[i]
                total_count += 1

        # 自环使用平均值
        if total_count != 0:
            avg_resistance = total_r / total_count
        else:
            avg_resistance = 1

        for i in range(sub_adj.shape[0]):
            edge_str_1 = str(i) + ',' + str(i)
            Effective_result[edge_str_1] = avg_resistance

        return Effective_result

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
                from baselines.dig.xgraph.method import DeepLIFT
                explainer = DeepLIFT(self.model_gnn, explain_graph=False)
                sparsity = 1
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        num_classes=self.num_classes, node_idx=node_idx_mapped,
                                        edge_weight=map_edge_weight)
                edge_mask = masks[predict_label]

            elif self.cfg.method == "flowx":
                from baselines.dig.xgraph.method import FlowMask
                explainer = FlowMask(self.model_gnn, explain_graph=False)
                sparsity = 0
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        num_classes=self.num_classes, node_idx=node_idx_mapped,
                                        edge_weight=map_edge_weight)
                edge_mask = masks[predict_label]

            elif self.cfg.method == "gnnlrp":
                from baselines.dig.xgraph.method import gnn_lrp
                explainer = GNN_LRP(self.model_gnn, explain_graph=False)
                sparsity = 0
                _, masks, _ = explainer(sub_features, map_edge_index, sparsity=sparsity,
                                        node_idx=node_idx_mapped, edge_weight=map_edge_weight,
                                        given_class=predict_label)
                edge_mask = masks[0]
                edge_mask = custom_log_transform(edge_mask)
            elif self.cfg.method == "convex":
                relu_delta, relu_end, relu_start = self.gen_data.gen_parameters_v2(
                    sub_features, map_edge_index, map_edge_weight, self.model_gnn
                )

                _, _, test_edge_result = test_path_contribution_edge(
                    paths,
                    csr_matrix(sub_adj.shape),
                    sub_adj,
                    target_edgelist,
                    relu_delta,
                    relu_start,
                    relu_end,
                    sub_features,
                    self.W1,
                    self.W2
                )

                target_edge_result = map_target(test_edge_result, node_idx_mapped)

                # 计算边的重要性分数
                select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))
                select_edges_list_value, _ = main_con_edge(
                    select_edge_important,
                    node_idx_mapped,
                    target_edge_result,
                    target_edgelist,
                    torch.zeros_like(output[node_idx_mapped]).numpy(),
                    output
                )

                edge_mask = torch.tensor(select_edges_list_value, dtype=torch.float32)

            elif self.cfg.method == "gnnexplainer":
                from baselines.torch_geometric.explain import GNNExplainer, Explainer

                explainer = Explainer(
                    model=self.model_gnn,
                    algorithm=GNNExplainer(epochs=200, lr=0.001),
                    explanation_type='model',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='node',
                        return_type='raw'
                    ),
                )

                explanation = explainer(sub_features, map_edge_index, index=node_idx_mapped,
                                        edge_weight=map_edge_weight)
                edge_mask = explanation.edge_mask

            elif self.cfg.method == "pgexplainer":
                from baselines.torch_geometric.explain import PGExplainer, Explainer

                train_epoch = 100
                train_lr = 0.001

                explainer = Explainer(
                    model=self.model_gnn,
                    algorithm=PGExplainer(epochs=train_epoch, lr=train_lr),
                    explanation_type='phenomenon',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='node',
                        return_type='raw'
                    ),
                )

                # PGExplainer 需要训练
                # 获取子图对应的标签
                sub_labels = self.labels[list(submapping.keys())]

                for epoch in range(train_epoch):
                    loss = explainer.algorithm.train(
                        epoch, self.model_gnn, sub_features, map_edge_index,
                        target=sub_labels, index=node_idx_mapped,
                        edge_weight=map_edge_weight
                    )

                explanation = explainer(sub_features, map_edge_index, index=node_idx_mapped,
                                        edge_weight=map_edge_weight, target=sub_labels)
                edge_mask = explanation.edge_mask






            # 后处理（如果是 GNN-LRP 则进行 log 变换）
            if self.cfg.method == "gnnlrp":
                edge_mask = custom_log_transform(edge_mask)

            # 保存
            np.save(save_path, edge_mask.detach().cpu().numpy())
            print(f"Node {node_idx}: Computed and saved")

        return edge_mask






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
    p.add_argument("--method", type=str, default="gnnexplainer",
                   choices=["deeplift", "flowx", "gnnexplainer", "gnnlrp", "pgexplainer",'convex'])

    p.add_argument("--sparsity", type=float, default=0.1,
                   help="select_edge_important = sparsity * total_edge")

    # Ricci
    p.add_argument("--ricci_alpha", type=float, default=0.0)
    p.add_argument("--ricci_lambda", type=float, default=0.0)

    # 采样
    p.add_argument("--sample_num_nodes", type=int, default=500)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--percentage_threshold", type=float, default=0.8)

    # 扰动
    p.add_argument("--num_remove_val_ratio", type=float, default=0.1)
    p.add_argument("--num_add_val_ratio", type=float, default=0.1)

    # 保存
    p.add_argument("--result_root", type=str, default="./result")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--normalize_adj",  default=True)  # 新增

    p.add_argument("--run_mode", type=str, default="select",
                   choices=["train", "val", "select"],
                   help="运行模式: train=筛选训练节点, val=筛选验证节点, select=优化lambda")

    # Optuna 配置
    p.add_argument("--n_trials", type=int, default=10, help="Optuna 试验次数")
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.1)
    p.add_argument("--max_train_nodes", type=int, default=250)

    p.add_argument("--curvature_type", type=str, default="resistance",
                   choices=["ricci", "resistance"],
                   help="曲率类型: ricci 或 resistance")
    p.add_argument("--resistance_epsilon", type=float, default=0.01,
                   help="Effective Resistance 计算的 epsilon 参数")
    p.add_argument("--resistance_method", type=str, default="kts",
                   help="Effective Resistance 计算方法")


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
        sparsity=args.sparsity,



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
        normalize_adj=args.normalize_adj,
        curvature_type=args.curvature_type,
        resistance_epsilon=args.resistance_epsilon,
        resistance_method=args.resistance_method,
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