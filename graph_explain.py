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
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from enum import Enum
import torch_geometric.transforms as T
import torch_geometric.nn as gnn
from torch_geometric.typing import Adj, OptTensor, Size
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils.loop import remove_self_loops
from scipy.sparse import csr_matrix


from utils.graph_utils import (initializeNodes,rumor_construct_adj_matrix,
                               rumor_construct_adj_matrix_v2,matrixtodict,dfs2,reverse_paths,
clear,softmax,custom_log_transform,edge_percentage,test_path_contribution_edge,main_con_edge,normalize_ricci,
KL_divergence
                               )
from utils.models_graph import GCN_Graph,GCN_Graph_pyg

class RunMode(Enum):
    TRAIN = "train"      # 筛选训练图
    VAL = "val"          # 筛选验证图
    SELECT = "select"    # Optuna 优化 lambda


@dataclass
class ExplainConfig:
    # 数据集配置
    dataset: str = "IMDB-BINARY"  # IMDB-BINARY, PROTEINS, MUTAG, etc.
    data_root: str = "./data/TUDataset"

    # 模型配置
    hidden_dim: int = 16
    nclass: int = 2
    dropout: float = 0.0
    layer_numbers: int = 2
    normalize_adj: bool = False

    # 解释方法
    method: str = "deeplift"  # deeplift, flowx, gnnexplainer, gnnlrp, pgexplainer
    run_mode: str = "select"

    curvature_type: str = "resistance"  # ricci 或 resistance

    sparsity: float = 0.1

    # Ricci 曲率配置
    ricci_alpha: float = 0.0
    ricci_lambda: float = 0.0

    # 图采样配置
    sample_num_graphs: int = 500
    train_ratio: float = 0.5
    percentage_threshold: float = 0.5
    resistance_epsilon: float = 0.01
    resistance_method: str = 'kts'

    n_trials: int = 50
    lambda_min: float = 0.0
    lambda_max: float = 0.2
    max_train_graphs: int = 250

    num_remove_val_ratio: float = 0.3
    num_add_val_ratio: float = 0.3

    # 保存路径
    result_root: str = "./result/"
    edge_mask_dir: str = "edge_masks"
    lambda_dir: str = "no_normalize/resistance"

    # 其他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GenGraphData:
    def __init__(self, cfg: ExplainConfig, dataset):
        self.cfg = cfg
        self.dataset = dataset

    def gen_original_edge(self, graph_idx):
        """生成单个图的边数据"""
        data = self.dataset[graph_idx]
        x, edge_index = data.x, data.edge_index

        edge_index_list = edge_index.numpy().tolist()
        # 添加自环
        for i in range(x.shape[0]):
            edge_index_list[0].append(i)
            edge_index_list[1].append(i)

        edge_index = torch.tensor(edge_index_list)

        if self.cfg.normalize_adj:
            adj_old = rumor_construct_adj_matrix(edge_index_list, x.shape[0])
        else:
            edge_weight = [1.0] * len(edge_index_list[0])
            adj_old = rumor_construct_adj_matrix_v2(edge_index_list, x.shape[0])

        adj_old_nonzero = adj_old.nonzero()
        graph_old = matrixtodict(adj_old_nonzero)

        edges_dict_old = dict()
        edges_weight_old = []
        for i, node in enumerate(edge_index_list[0]):
            key = str(node) + ',' + str(edge_index_list[1][i])
            edges_dict_old[key] = i
            edges_weight_old.append(adj_old[node, edge_index_list[1][i]])

        edges_weight_old = torch.tensor(edges_weight_old, dtype=torch.float32)

        return x, edge_index, graph_old, edges_dict_old, adj_old, edges_weight_old

    def gen_parameters_v2(self, model, features, edges_tensor, edgeweight):
        """计算 convex 方法所需的 relu 参数"""
        model.eval()
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_tensor, edgeweight)
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

        W1 = model.state_dict()['conv1.lin.weight'].t()
        W2 = model.state_dict()['conv2.lin.weight'].t()

        return W1, W2, relu_delta, relu_end, relu_start

class GraphExplainer:
    def __init__(self, cfg: ExplainConfig):
        self.cfg = cfg
        self.setup_seed()
        self.load_dataset()
        self.load_models()
        self.gen_data = GenGraphData(cfg, self.dataset)
        self._init_data_split()

    def _init_data_split(self):
        """初始化训练/验证数据集划分"""
        all_graphs = list(range(len(self.dataset)))
        target_graph_list = random.sample(
            all_graphs,
            min(self.cfg.sample_num_graphs, len(all_graphs))
        )

        split_idx = int(self.cfg.train_ratio * len(target_graph_list))
        self.train_graphs = target_graph_list[:split_idx]
        self.val_graphs = target_graph_list[split_idx:]

        print(f"Data split: train={len(self.train_graphs)}, val={len(self.val_graphs)}")

    def get_lambda_save_dir(self) -> str:
        """获取 lambda 相关文件的保存目录"""
        norm_dir = "normalize" if self.cfg.normalize_adj else "no_normalize"
        save_dir = os.path.join(
            self.cfg.result_root,
            norm_dir,
            self.cfg.curvature_type,
            self.modelname,
            self.cfg.method
        )
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def verify_train_graph(self, graph_idx: int) -> float:
        """验证图是否适合训练"""
        edge_mask = self.explain_graph(graph_idx, use_cache=True)
        if edge_mask is None:
            return 1.0
        percentage = edge_percentage(edge_mask)
        return percentage

    def run_train_mode(self):
        """筛选训练图并保存"""
        print(f"[TRAIN MODE] Filtering {len(self.train_graphs)} graphs...")

        valid_graphs = []
        for graph_idx in self.train_graphs:
            percentage = self.verify_train_graph(graph_idx)
            if percentage < self.cfg.percentage_threshold:
                valid_graphs.append(graph_idx)
                print(f"Graph {graph_idx}: VALID (percentage={percentage:.4f})")
            else:
                print(f"Graph {graph_idx}: SKIP (percentage={percentage:.4f})")

        save_dir = self.get_lambda_save_dir()
        save_path = os.path.join(save_dir, "train_node.json")
        with open(save_path, 'w') as f:
            json.dump(valid_graphs, f)

        print(f"[TRAIN MODE] Saved {len(valid_graphs)} graphs to {save_path}")
        return valid_graphs

    def run_val_mode(self):
        """筛选验证图并保存"""
        save_dir = self.get_lambda_save_dir()

        val_graph_path = os.path.join(save_dir, "val_node.json")
        if not os.path.exists(val_graph_path):
            print(f"[VAL MODE] val_node.json not found, generating...")
            valid_graphs = []
            for graph_idx in self.val_graphs:
                percentage = self.verify_train_graph(graph_idx)
                if percentage < self.cfg.percentage_threshold:
                    valid_graphs.append(graph_idx)
                    print(f"Graph {graph_idx}: VALID (percentage={percentage:.4f})")

            with open(val_graph_path, 'w') as f:
                json.dump(valid_graphs, f)
        else:
            with open(val_graph_path, 'r') as f:
                valid_graphs = json.load(f)

        print(f"[VAL MODE] Loaded {len(valid_graphs)} validation graphs")

        good_lambdas_path = os.path.join(save_dir, "good_lambdas.json")
        if not os.path.exists(good_lambdas_path):
            print(f"[VAL MODE] good_lambdas.json not found, please run SELECT mode first")
            return None

        with open(good_lambdas_path, 'r') as f:
            good_lambdas_list = json.load(f)

        print(f"[VAL MODE] Loaded {len(good_lambdas_list)} good lambdas: {good_lambdas_list}")

        for graph_idx in valid_graphs:
            print(f"[VAL MODE] Evaluating graph {graph_idx}...")
            self.evaluate_val_graph(graph_idx, good_lambdas_list)

        print(f"[VAL MODE] Completed evaluation for {len(valid_graphs)} graphs")
        return valid_graphs

    def run_select_mode(self):
        """使用 Optuna 优化 lambda 参数"""
        save_dir = self.get_lambda_save_dir()
        train_graph_path = os.path.join(save_dir, "train_node.json")

        if not os.path.exists(train_graph_path):
            print(f"[SELECT MODE] train_node.json not found, running TRAIN mode first...")
            self.run_train_mode()

        with open(train_graph_path, 'r') as f:
            target_graphs = json.load(f)

        if len(target_graphs) > self.cfg.max_train_graphs:
            target_graphs = random.sample(target_graphs, self.cfg.max_train_graphs)

        print(f"[SELECT MODE] Optimizing lambda with {len(target_graphs)} graphs...")

        good_lambdas = []

        def objective(trial):
            lam = trial.suggest_float("lambda", self.cfg.lambda_min, self.cfg.lambda_max)

            probs = []
            for graph_idx in target_graphs:
                prob = self.evaluate_graph_with_lambda(graph_idx, lam)
                if prob is not None:
                    probs.append(prob)

            if len(probs) == 0:
                return 0.0

            avg_prob = np.mean(probs)

            if avg_prob < 0:
                good_lambdas.append(lam)
                print(f"Lambda {lam:.4f}: avg_prob={avg_prob:.4f} (GOOD)")

                with open(os.path.join(save_dir, "good_lambdas.json"), 'w') as f:
                    json.dump(good_lambdas, f)

            return avg_prob

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
                       subadj, num_val, all_edges, sub_features, output_goal, model):
        """评估选定边的鲁棒性"""
        result_dict = dict()
        result_dict_KL = dict()

        predict_old_label = np.argmax(softmax(output_goal.view(-1).detach().numpy()))

        edges_old_dict = dict()
        for i in range(len(edges_index[0])):
            edges_old_dict[str(edges_index[0][i].item()) + ',' + str(edges_index[1][i].item())] = i

        edges_old_dict_reverse = dict()
        for key, value in edges_old_dict.items():
            node_list = key.split(',')
            edges_old_dict_reverse[value] = [int(node_list[0]), int(node_list[1])]

        add_bound = subadj.shape[0] * (subadj.shape[0] + 1) / 2 - all_edges
        num_add_actual = min(num_add, add_bound)

        for count in range(num_val):
            tmp_changed_edgelist = []

            edges_weight_new = copy.deepcopy(edges_weight).tolist()
            edges_index_new = copy.deepcopy(edges_index).tolist()

            # 随机移除边
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

            # 随机添加边
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

            tmp_output = model.verify_layeredge(
                sub_features,
                torch.tensor(edges_index_new),
                torch.tensor(edges_index_new),
                edge_weight1=torch.tensor(edges_weight_new, dtype=torch.float32),
                edge_weight2=torch.tensor(edges_weight_new, dtype=torch.float32)
            ).view(-1)

            result_dict[count] = float(abs(
                softmax(tmp_output.detach().numpy())[predict_old_label] -
                softmax(output_goal.view(-1).detach().numpy())[predict_old_label]
            ))

            result_dict_KL[count] = KL_divergence(
                softmax(output_goal.view(-1).detach().numpy()),
                softmax(tmp_output.detach().numpy())
            )

        return result_dict, result_dict_KL

    def ave(self, my_dict):
        """计算字典值的平均"""
        non_zero_values = [v for v in my_dict.values() if v != float('inf')]
        if non_zero_values:
            return sum(non_zero_values) / len(non_zero_values), len(non_zero_values)
        return 0, 0

    def evaluate_graph_with_lambda(self, graph_idx: int, lam: float) -> Optional[float]:
        """评估图在给定 lambda 下的鲁棒性差值"""
        edge_mask = self.explain_graph(graph_idx, use_cache=True)
        if edge_mask is None:
            return None

        x, edge_index, graph, edges_dict, adj, edges_weight = self.gen_data.gen_original_edge(graph_idx)
        x = x.to(torch.float32)

        # 获取路径和目标边
        path_ceshi = []
        for change_node in range(x.shape[0]):
            old_paths = dfs2(change_node, change_node, graph, self.cfg.layer_numbers + 1, [], [])
            path_ceshi = path_ceshi + old_paths

        target_path = reverse_paths(path_ceshi)

        target_edgelist = []
        for path in target_path:
            if [path[0], path[1]] not in target_edgelist and [path[1], path[0]] not in target_edgelist:
                target_edgelist.append([path[0], path[1]])
            if [path[2], path[1]] not in target_edgelist and [path[1], path[2]] not in target_edgelist:
                target_edgelist.append([path[1], path[2]])
        target_edgelist = clear(target_edgelist)

        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))

        if len(target_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            return None

        # 模型输出
        output_old = self.model.forward(x, edge_index, edges_weight).view(-1)
        predict_old_label = np.argmax(softmax(output_old.detach().numpy()))

        # 构建边索引和权重
        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in target_edgelist:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(adj[edge[0], edge[1]])
                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(adj[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(adj[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight, dtype=torch.float32)

        # 计算曲率
        curvature_result = self.compute_curvature(edge_index, edges_weight, adj)

        select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))
        if select_edge_important < 1:
            return None

        # 基础选择（不使用曲率）

        if self.cfg.method == "convex":
            # convex 方法: edge_mask 对应 target_edgelist
            total_result_base = dict()
            for i, edge in enumerate(target_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                if i < len(edge_mask):
                    total_result_base[edge_str] = edge_mask[i].item()
        else:
            # 其他方法: edge_mask 对应 edge_index
            total_result_base = dict()
            for i in range(len(edge_index[0])):
                edge_str = str(edge_index[0][i].item()) + ',' + str(edge_index[1][i].item())
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

        # 曲率增强选择

        if self.cfg.method == "convex":
            total_result_curvature = dict()
            for i, edge in enumerate(target_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                curvature_val = curvature_result.get(edge_str, 0)
                if i < len(edge_mask):
                    total_result_curvature[edge_str] = self.compute_combined_score(
                        edge_mask[i].item(), curvature_val, lam
                    )
        else:
            total_result_curvature = dict()
            for i in range(len(edge_index[0])):
                edge_str = str(edge_index[0][i].item()) + ',' + str(edge_index[1][i].item())
                curvature_val = curvature_result.get(edge_str, 0)
                total_result_curvature[edge_str] = self.compute_combined_score(
                    edge_mask[i].item(), curvature_val, lam
                )


        sort_diff_curv = sorted(total_result_curvature.items(), key=lambda x: x[1], reverse=True)

        select_gnn_edgelist = []
        select_idx = 0
        while len(select_gnn_edgelist) < select_edge_important and select_idx < len(sort_diff_curv):
            edge_str = sort_diff_curv[select_idx][0]
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

        if select_flag:
            return 0

        # 评估鲁棒性
        num_val = 100
        result_val_prob_curv, _ = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
            num_add_val, num_remove_val, adj, num_val, len(target_edgelist),
            x, output_old.view(1, -1), self.model
        )
        prob_robust, _ = self.ave(result_val_prob_curv)

        result_val_prob_base, _ = self.val_robustness(
            target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
            num_add_val, num_remove_val, adj, num_val, len(target_edgelist),
            x, output_old.view(1, -1), self.model
        )
        prob_robust_base, _ = self.ave(result_val_prob_base)

        return prob_robust - prob_robust_base

    def evaluate_val_graph(self, graph_idx: int, good_lambdas_list: list):
        """评估验证图"""
        x, edge_index, graph, edges_dict, adj, edges_weight = self.gen_data.gen_original_edge(graph_idx)
        x = x.to(torch.float32)

        # 获取路径和目标边
        path_ceshi = []
        for change_node in range(x.shape[0]):
            old_paths = dfs2(change_node, change_node, graph, self.cfg.layer_numbers + 1, [], [])
            path_ceshi = path_ceshi + old_paths

        target_path = reverse_paths(path_ceshi)

        target_edgelist = []
        for path in target_path:
            if [path[0], path[1]] not in target_edgelist and [path[1], path[0]] not in target_edgelist:
                target_edgelist.append([path[0], path[1]])
            if [path[2], path[1]] not in target_edgelist and [path[1], path[2]] not in target_edgelist:
                target_edgelist.append([path[1], path[2]])
        target_edgelist = clear(target_edgelist)

        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))

        if len(target_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Graph {graph_idx}: Skipped (edges={len(target_edgelist)})")
            return

        output_old = self.model.forward(x, edge_index, edges_weight)
        predict_old_label = np.argmax(softmax(output_old.view(-1).detach().numpy()))

        curvature_result = self.compute_curvature(edge_index, edges_weight, adj)
        edge_mask = self.explain_graph(graph_idx, use_cache=True)
        if edge_mask is None:
            return

        select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))

        # 构建边索引和权重
        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in target_edgelist:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(adj[edge[0], edge[1]])
                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(adj[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(adj[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight, dtype=torch.float32)

        for lam in good_lambdas_list:
            result_important_dict = dict()
            result_important_base_dict = dict()
            save_flag = False

            # 基础选择
            total_result_base = dict()
            for i in range(len(edge_index[0])):
                edge_str = str(edge_index[0][i].item()) + ',' + str(edge_index[1][i].item())
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

            # 曲率增强选择
            total_result_curvature = dict()
            for i in range(len(edge_index[0])):
                edge_str = str(edge_index[0][i].item()) + ',' + str(edge_index[1][i].item())
                curvature_val = curvature_result.get(edge_str, 0)
                total_result_curvature[edge_str] = self.compute_combined_score(
                    edge_mask[i].item(), curvature_val, lam
                )

            sort_diff_curv = sorted(total_result_curvature.items(), key=lambda x: x[1], reverse=True)

            select_gnn_edgelist = []
            select_idx = 0
            while len(select_gnn_edgelist) < select_edge_important and select_idx < len(sort_diff_curv):
                edge_str = sort_diff_curv[select_idx][0]
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
                num_val = 100

                # 评估曲率增强选择
                result_val_prob, result_val_kl = self.val_robustness(
                    target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
                    num_add_val, num_remove_val, adj, num_val, len(target_edgelist),
                    x, output_old, self.model
                )

                result_important_dict[f'0,select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])] for e in select_gnn_edgelist]
                result_important_dict[f'0,robustness {self.cfg.method} prob'] = result_val_prob
                result_important_dict[f'0,robustness {self.cfg.method} kl'] = result_val_kl

                # 评估基础选择
                result_val_prob_base, result_val_kl_base = self.val_robustness(
                    target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
                    num_add_val, num_remove_val, adj, num_val, len(target_edgelist),
                    x, output_old, self.model
                )

                result_important_base_dict[f'0,select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])] for e in select_gnn_base_edgelist]
                result_important_base_dict[f'0,robustness {self.cfg.method} prob'] = result_val_prob_base
                result_important_base_dict[f'0,robustness {self.cfg.method} kl'] = result_val_kl_base

            # 保存结果
            if save_flag:
                norm_dir = "normalize" if self.cfg.normalize_adj else "no_normalize"
                curv_save_dir = os.path.join(
                    self.cfg.result_root, norm_dir,
                    self.cfg.curvature_type, self.modelname, self.cfg.method,
                    f"{self.cfg.num_add_val_ratio}_{self.cfg.num_remove_val_ratio}",
                    str(lam), self.cfg.curvature_type
                )
                os.makedirs(curv_save_dir, exist_ok=True)

                with open(os.path.join(curv_save_dir, f"{graph_idx}.json"), 'w') as f:
                    json.dump(result_important_dict, f)

                base_save_dir = os.path.join(
                    self.cfg.result_root, norm_dir,
                    self.cfg.curvature_type, self.modelname, self.cfg.method,
                    f"{self.cfg.num_add_val_ratio}_{self.cfg.num_remove_val_ratio}",
                    str(lam), 'base'
                )
                os.makedirs(base_save_dir, exist_ok=True)

                with open(os.path.join(base_save_dir, f"{graph_idx}.json"), 'w') as f:
                    json.dump(result_important_base_dict, f)

                print(f"Graph {graph_idx}, Lambda {lam}: Results saved")

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
        self.dataset = TUDataset(self.cfg.data_root, name=self.cfg.dataset, use_node_attr=True)
        initializeNodes(self.dataset)

        self.modelname = self.cfg.dataset
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes

        print(f"Loaded dataset: {self.cfg.dataset}")
        print(f"  Graphs: {len(self.dataset)}, Features: {self.num_features}, Classes: {self.num_classes}")

    def load_models(self):
        """加载预训练模型"""
        if self.cfg.normalize_adj:
            model_path = f'./checkpoints/graph_{self.modelname}_norm.pt'

        else:
            model_path = f'./checkpoints/graph_{self.modelname}_nonorm.pt'

        print('model_path', model_path)

        if self.cfg.method in ["deeplift", "flowx", "gnnlrp", "convex"]:
            # DIG 方法使用自定义 GCNConv（支持 __explain_flow__）
            self.model = GCN_Graph(
                nfeat=self.num_features,
                hidden_channels=self.cfg.hidden_dim,
                nclass=self.num_classes
            )
        else:
            # PyG 方法（gnnexplainer, pgexplainer）使用标准 GCNConv
            self.model = GCN_Graph_pyg(
                nfeat=self.num_features,
                hidden_channels=self.cfg.hidden_dim,
                nclass=self.num_classes
            )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')

            # 从 checkpoint 中提取真正的模型权重
            if 'state' in checkpoint:
                state_dict = checkpoint['state']
            else:
                state_dict = checkpoint

            model_dict = dict()

            if 'gc1.weight' in state_dict:
                # 格式 1: gc1.weight, gc2.weight
                src_conv1 = 'gc1.weight'
                src_conv2 = 'gc2.weight'
                need_transpose = True
            elif 'conv1.lin.weight' in state_dict:
                # 格式 2: conv1.lin.weight, conv2.lin.weight
                src_conv1 = 'conv1.lin.weight'
                src_conv2 = 'conv2.lin.weight'
                need_transpose = False
            elif 'conv1.weight' in state_dict:
                # 格式 3: conv1.weight, conv2.weight
                src_conv1 = 'conv1.weight'
                src_conv2 = 'conv2.weight'
                need_transpose = False
            else:
                raise KeyError(f"Unknown state_dict format. Keys: {state_dict.keys()}")

            if self.cfg.method in ["deeplift", "flowx", "gnnlrp", "convex"]:
                # DIG 方法需要 conv1.weight 和 conv1.lin.weight
                if need_transpose:
                    model_dict['conv1.weight'] = state_dict[src_conv1]
                    model_dict['conv2.weight'] = state_dict[src_conv2]
                    model_dict['conv1.lin.weight'] = state_dict[src_conv1].t()
                    model_dict['conv2.lin.weight'] = state_dict[src_conv2].t()
                else:
                    model_dict['conv1.lin.weight'] = state_dict[src_conv1]
                    model_dict['conv2.lin.weight'] = state_dict[src_conv2]
                    model_dict['conv1.weight'] = state_dict[src_conv1].t()
                    model_dict['conv2.weight'] = state_dict[src_conv2].t()

                self.model.load_state_dict(model_dict)
                self.W1 = self.model.state_dict()['conv1.lin.weight'].t()
                self.W2 = self.model.state_dict()['conv2.lin.weight'].t()
            else:
                # PyG 方法直接加载权重
                model_dict = dict()
                if 'gc1.weight' in state_dict:
                    # 旧格式: gc1.weight -> conv1.lin.weight (需要转置)
                    model_dict['conv1.lin.weight'] = state_dict['gc1.weight'].t()
                    model_dict['conv2.lin.weight'] = state_dict['gc2.weight'].t()
                else:
                    # 新格式: 直接使用
                    model_dict['conv1.lin.weight'] = state_dict['conv1.lin.weight']
                    model_dict['conv2.lin.weight'] = state_dict['conv2.lin.weight']

                self.model.load_state_dict(model_dict)
                self.model.eval()

            print(f"Loaded model weights from {model_path}")
            print(
                f"  Using model: {'GCN_Graph (DIG)' if self.cfg.method in ['deeplift', 'flowx', 'gnnlrp', 'convex'] else 'GCN_Graph_pyg (PyG)'}")
        else:
            print(f"Warning: Model file not found: {model_path}")





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
        """计算边的组合分数"""
        if self.cfg.curvature_type == "ricci":
            return edge_mask_value + lam * curvature_value
        elif self.cfg.curvature_type == "resistance":
            return edge_mask_value - lam * curvature_value
        return edge_mask_value

    def compute_curvature(self, edge_index, edge_weight, adj) -> dict:
        """根据配置计算曲率"""
        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if isinstance(edge_weight, list):
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        if self.cfg.curvature_type == "ricci":
            return self.compute_ricci_curvature(edge_index, adj)
        elif self.cfg.curvature_type == "resistance":
            return self.compute_resistance(edge_index, edge_weight, adj)
        else:
            raise ValueError(f"Unknown curvature type: {self.cfg.curvature_type}")

    def compute_ricci_curvature(self, edge_index, adj) -> dict:
        """计算 Ollivier-Ricci 曲率"""
        if isinstance(edge_index, torch.Tensor):
            edge_index_tensor = edge_index
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

        old_data = Data(edge_index=edge_index_tensor, num_nodes=adj.shape[0])
        G_old = to_networkx(old_data, to_undirected=True)

        orc = OllivierRicci(G_old, alpha=self.cfg.ricci_alpha)
        orc.compute_ricci_curvature()

        ricci_result = dict()
        for node_1, node_2 in orc.G.edges:
            curv = normalize_ricci(orc.G[node_1][node_2]['ricciCurvature'])
            ricci_result[f"{node_1},{node_2}"] = curv
            ricci_result[f"{node_2},{node_1}"] = curv

        median_number = sum(ricci_result.values()) / len(ricci_result) if ricci_result else 0
        for i in range(adj.shape[0]):
            ricci_result[f"{i},{i}"] = median_number

        return ricci_result

    def compute_resistance(self, edge_index, edge_weight, adj) -> dict:
        """计算 Effective Resistance"""
        from utils.Network import Network

        if isinstance(edge_index, list):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if isinstance(edge_weight, list):
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        edge_index_remove, edge_weight_remove = remove_self_loops(edge_index, edge_weight)

        old_data_remove = Data(edge_index=edge_index_remove, edge_weight=edge_weight_remove,
                               num_nodes=adj.shape[0])
        G_old = to_networkx(old_data_remove, to_undirected=True)

        network = Network(None, None, G_old)
        E_list, Effective_R = network.effR(self.cfg.resistance_epsilon, self.cfg.resistance_method)

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

        avg_resistance = total_r / total_count if total_count != 0 else 1

        for i in range(adj.shape[0]):
            Effective_result[f"{i},{i}"] = avg_resistance

        return Effective_result

    def explain_graph(self, graph_idx: int, use_cache: bool = True) -> Optional[torch.Tensor]:
        """对单个图进行解释"""
        x, edge_index, graph, edges_dict, adj, edges_weight = self.gen_data.gen_original_edge(graph_idx)
        x = x.to(torch.float32)

        # 获取路径和目标边
        path_ceshi = []
        for change_node in range(x.shape[0]):
            old_paths = dfs2(change_node, change_node, graph, self.cfg.layer_numbers + 1, [], [])
            path_ceshi = path_ceshi + old_paths

        target_path = reverse_paths(path_ceshi)

        target_edgelist = []
        for path in target_path:
            if [path[0], path[1]] not in target_edgelist and [path[1], path[0]] not in target_edgelist:
                target_edgelist.append([path[0], path[1]])
            if [path[2], path[1]] not in target_edgelist and [path[1], path[2]] not in target_edgelist:
                target_edgelist.append([path[1], path[2]])
        target_edgelist = clear(target_edgelist)

        num_remove_val = math.floor(self.cfg.num_remove_val_ratio * len(target_edgelist))
        num_add_val = math.floor(self.cfg.num_add_val_ratio * len(target_edgelist))

        if len(target_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Graph {graph_idx}: Skipped (edges={len(target_edgelist)})")
            return None

        output_old = self.model.forward(x, edge_index, edges_weight).view(-1)
        predict_old_label = np.argmax(softmax(output_old.detach().numpy()))

        save_dir = self.get_save_dir()
        if self.cfg.method == "convex":
            save_path = os.path.join(save_dir, f"{self.cfg.method}_{graph_idx}.json")
        else:
            save_path = os.path.join(save_dir, f"{self.cfg.method}_{graph_idx}.npy")

        if use_cache and os.path.exists(save_path):
            if self.cfg.method == "convex":
                with open(save_path, 'r') as f:
                    select_edges_list_value = json.load(f)
                edge_mask = torch.tensor(select_edges_list_value, dtype=torch.float32)
            else:
                edge_mask_array = np.load(save_path)
                edge_mask = torch.tensor(edge_mask_array, dtype=torch.float32)
            print(f"Graph {graph_idx}: Loaded from cache")
        else:
            if self.cfg.method == "deeplift":
                from baselines.dig.xgraph.method import DeepLIFT
                explainer = DeepLIFT(self.model, explain_graph=True)
                sparsity = 0
                masks, _, _ = explainer(x, edge_index, sparsity=sparsity, num_classes=self.cfg.nclass,
                                        edge_weight=edges_weight)
                edge_mask = masks[predict_old_label]
                edge_mask = custom_log_transform(edge_mask)

            elif self.cfg.method == "flowx":
                from baselines.dig.xgraph.method import FlowMask
                explainer = FlowMask(self.model, explain_graph=True)
                sparsity = 0
                _, masks, _ = explainer(x, edge_index, sparsity=sparsity, num_classes=self.cfg.nclass,
                                        edge_weight=edges_weight)
                edge_mask = masks[predict_old_label]

            elif self.cfg.method == "gnnlrp":
                from baselines.dig.xgraph.method import GNN_LRP
                explainer = GNN_LRP(self.model, explain_graph=True)
                sparsity = 0
                _, masks, _ = explainer(x, edge_index, sparsity=sparsity, edge_weight=edges_weight,
                                        given_class=predict_old_label)
                edge_mask = masks[0]
                edge_mask = custom_log_transform(edge_mask)

            elif self.cfg.method=='convex':
                logit_old = self.model.pre_forward(x, edge_index, edges_weight)

                W1, W2, relu_delta, relu_end, relu_start = self.gen_data.gen_parameters_v2(
                    self.model, x, edge_index, edges_weight
                )

                # 计算边贡献
                _, _, test_edge_result = test_path_contribution_edge(
                    target_path,
                    csr_matrix(adj.shape),
                    adj,
                    target_edgelist,
                    relu_delta,
                    relu_start,
                    relu_end,
                    x,
                    W1,
                    W2
                )

                # 计算选择的边数量
                select_edge_important = math.ceil(self.cfg.sparsity * len(target_edgelist))

                # 使用凸优化计算边的重要性分数
                select_edges_list_value, select_edges_list_sort = main_con_edge(
                    select_edge_important,
                    test_edge_result,
                    target_edgelist,
                    torch.zeros_like(logit_old.detach()),
                    output_old.detach().numpy()
                )

                edge_mask = torch.tensor(select_edges_list_value, dtype=torch.float32)

                # 保存为 json 格式
                with open(save_path, 'w') as f:
                    json.dump(select_edges_list_value, f)
                print(f"Graph {graph_idx}: Computed and saved")
                return edge_mask

            elif self.cfg.method == "gnnexplainer":
                from baselines.torch_geometric.explain import GNNExplainer, Explainer

                explainer = Explainer(
                    model=self.model,
                    algorithm=GNNExplainer(epochs=200, lr=0.001),
                    explanation_type='model',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='graph',
                        return_type='raw'
                    ),
                )

                explanation = explainer(x, edge_index, edge_weight=edges_weight)
                edge_mask = explanation.edge_mask

            elif self.cfg.method == "pgexplainer":
                from baselines.torch_geometric.explain import PGExplainer, Explainer

                train_epoch = 100
                train_lr = 0.001

                explainer = Explainer(
                    model=self.model,
                    algorithm=PGExplainer(epochs=train_epoch, lr=train_lr),
                    explanation_type='phenomenon',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='graph',
                        return_type='raw'
                    ),
                )

                data = self.dataset[graph_idx]
                target = data.y

                for epoch in range(train_epoch):
                    loss = explainer.algorithm.train(
                        epoch, self.model, x, edge_index,
                        target=target, edge_weight=edges_weight
                    )

                explanation = explainer(x, edge_index, edge_weight=edges_weight, target=target)
                edge_mask = explanation.edge_mask


            else:
                raise ValueError(f"Unknown method: {self.cfg.method}")

            # 保存
            np.save(save_path, edge_mask.detach().cpu().numpy())
            print(f"Graph {graph_idx}: Computed and saved")

        return edge_mask

def parse_args() -> ExplainConfig:
    p = argparse.ArgumentParser(description="Unified Graph Explanation with Curvature")

    p.add_argument("--dataset", type=str, default="IMDB-BINARY")
    p.add_argument("--data_root", type=str, default="./data")

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_numbers", type=int, default=2)

    p.add_argument("--method", type=str, default="gnnexplainer",
                   choices=["deeplift", "flowx", "gnnexplainer", "gnnlrp", "pgexplainer",'convex'])

    p.add_argument("--sparsity", type=float, default=0.1)

    p.add_argument("--ricci_alpha", type=float, default=0.0)
    p.add_argument("--ricci_lambda", type=float, default=0.0)

    p.add_argument("--sample_num_graphs", type=int, default=100)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--percentage_threshold", type=float, default=0.9)

    p.add_argument("--num_remove_val_ratio", type=float, default=0.1)
    p.add_argument("--num_add_val_ratio", type=float, default=0.1)

    p.add_argument("--result_root", type=str, default="./result")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--normalize_adj", action='store_true', default=False)

    p.add_argument("--run_mode", type=str, default="select",
                   choices=["train", "val", "select"])

    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.2)
    p.add_argument("--max_train_graphs", type=int, default=250)

    p.add_argument("--curvature_type", type=str, default="ricci",
                   choices=["ricci", "resistance"])
    p.add_argument("--resistance_epsilon", type=float, default=0.01)
    p.add_argument("--resistance_method", type=str, default="kts")

    args = p.parse_args()

    return ExplainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        hidden_dim=args.hidden_dim,
        nclass=args.nclass,
        dropout=args.dropout,
        layer_numbers=args.layer_numbers,
        method=args.method,
        ricci_alpha=args.ricci_alpha,
        ricci_lambda=args.ricci_lambda,
        sample_num_graphs=args.sample_num_graphs,
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
        max_train_graphs=args.max_train_graphs,
        normalize_adj=args.normalize_adj,
        curvature_type=args.curvature_type,
        resistance_epsilon=args.resistance_epsilon,
        resistance_method=args.resistance_method,
    )


# ========== 主函数 ==========
def main():
    cfg = parse_args()
    print(f"Config: {cfg}")

    explainer = GraphExplainer(cfg)

    results = explainer.run()


if __name__ == "__main__":
    main()