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
import copy
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
from utils.link_utils import (
    SynGraphDataset, split_edge, clear_time, clear_time_UCI,
    gen_link_data,softmax,dfs2,reverse_paths,clear,edge_percentage,test_path_contribution_edge,
map_target,mlp_contribution,main_con_edge,normalize_ricci,KL_divergence
)
from utils.Network import Network
from torch_geometric.utils.loop import remove_self_loops
from utils.models_link import NetLinkEvaluate,NetLinkEvaluatePYG
from scipy.sparse import csr_matrix
class RunMode(Enum):
    TRAIN = "train"      # 筛选训练节点
    VAL = "val"          # 筛选验证节点
    SELECT = "select"    # Optuna 优化 lambda

# ========== 配置类 ==========
@dataclass
class ExplainConfig:
    # 数据集配置
    dataset: str = "UCI" # UCI, bitcoinalpha, bitcoinotc
    data_root: str = "./data"

    start_time: int = 0
    end_time: int = 40
    time_flag: str = "week"  # "week" for UCI, "month" for bitcoin

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
    sample_num_edges: int = 500
    train_ratio: float = 0.5
    percentage_threshold: float = 0.8
    resistance_epsilon:float=0.01
    resistance_method: str='kts'

    n_trials: int = 50
    lambda_min: float = 0.0
    lambda_max: float = 0.1
    max_train_edges: int = 250


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
class LinkExplainer:
    def __init__(self, cfg: ExplainConfig):
        self.cfg = cfg
        self.setup_seed()
        self.load_dataset()
        self.load_models()

        self.gen_data = GenData(cfg)

        self._init_data_split()

    def _init_data_split(self):
        """初始化训练/验证数据集划分"""
        all_edges = self.target_edges_list.copy()
        random.shuffle(all_edges)

        sample_size = min(self.cfg.sample_num_edges, len(all_edges))
        sampled_edges = all_edges[:sample_size]

        split_idx = int(self.cfg.train_ratio * len(sampled_edges))
        self.train_edges = sampled_edges[:split_idx]
        self.val_edges = sampled_edges[split_idx:]

        print(f"Data split: train={len(self.train_edges)}, val={len(self.val_edges)}")

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

    def verify_train_edge(self, target_edge: list) -> float:
        """
        验证边是否适合训练
        返回 edge_percentage，值越小说明边重要性分布越不均匀，边越适合训练
        """
        edge_mask = self.explain_edge(target_edge, use_cache=True)
        if edge_mask is None:
            return 1.0  # 返回高值表示不适合
        print('edge_mask', edge_mask)
        percentage = edge_percentage(edge_mask)
        return percentage

    def run_train_mode(self):
        """筛选训练边并保存"""
        print(f"[TRAIN MODE] Filtering {len(self.train_edges)} edges...")

        valid_edges = []
        for target_edge in self.train_edges:
            percentage = self.verify_train_edge(target_edge)
            if percentage < self.cfg.percentage_threshold:
                valid_edges.append(target_edge)
                print(f"Edge {target_edge}: VALID (percentage={percentage:.4f})")
            else:
                print(f"Edge {target_edge}: SKIP (percentage={percentage:.4f})")

        # 保存训练边
        save_dir = self.get_lambda_save_dir()
        save_path = os.path.join(save_dir, "train_edge.json")  # 改为 train_edge.json
        with open(save_path, 'w') as f:
            json.dump(valid_edges, f)

        print(f"[TRAIN MODE] Saved {len(valid_edges)} edges to {save_path}")
        return valid_edges

    def run_val_mode(self):
        """筛选验证边并保存"""
        save_dir = self.get_lambda_save_dir()

        val_edge_path = os.path.join(save_dir, "val_edge.json")  # 改为 val_edge.json
        if not os.path.exists(val_edge_path):
            print(f"[VAL MODE] val_edge.json not found, generating...")
            valid_edges = []
            for target_edge in self.val_edges:
                percentage = self.verify_train_edge(target_edge)
                if percentage < self.cfg.percentage_threshold:
                    valid_edges.append(target_edge)
                    print(f"Edge {target_edge}: VALID (percentage={percentage:.4f})")

            with open(val_edge_path, 'w') as f:
                json.dump(valid_edges, f)
        else:
            with open(val_edge_path, 'r') as f:
                valid_edges = json.load(f)

        print(f"[VAL MODE] Loaded {len(valid_edges)} validation edges")

        good_lambdas_path = os.path.join(save_dir, "good_lambdas.json")
        if not os.path.exists(good_lambdas_path):
            print(f"[VAL MODE] good_lambdas.json not found, please run SELECT mode first")
            return None

        with open(good_lambdas_path, 'r') as f:
            good_lambdas_list = json.load(f)

        print(f"[VAL MODE] Loaded {len(good_lambdas_list)} good lambdas: {good_lambdas_list}")

        for target_edge in valid_edges:
            print(f"[VAL MODE] Evaluating edge {target_edge}...")
            self.evaluate_val_edge(target_edge, good_lambdas_list)

        print(f"[VAL MODE] Completed evaluation for {len(valid_edges)} edges")
        return valid_edges

    def run_select_mode(self):
        """使用 Optuna 优化 lambda 参数"""
        save_dir = self.get_lambda_save_dir()
        train_edge_path = os.path.join(save_dir, "train_edge.json")  # 改为 train_edge.json

        # 加载训练边
        if not os.path.exists(train_edge_path):
            print(f"[SELECT MODE] train_edge.json not found, running TRAIN mode first...")
            self.run_train_mode()

        with open(train_edge_path, 'r') as f:
            target_edges = json.load(f)

        # 限制边数量
        if len(target_edges) > self.cfg.max_train_edges:  # 改为 max_train_edges
            target_edges = random.sample(target_edges, self.cfg.max_train_edges)

        print(f"[SELECT MODE] Optimizing lambda with {len(target_edges)} edges...")

        good_lambdas = []

        def objective(trial):
            lam = trial.suggest_float("lambda", self.cfg.lambda_min, self.cfg.lambda_max)

            # 计算所有边的 prob_robust 差值
            probs = []
            for target_edge in target_edges:
                prob = self.evaluate_edge_with_lambda(target_edge, lam)  # 改为 evaluate_edge_with_lambda
                if prob is not None:
                    probs.append(prob)

            if len(probs) == 0:
                return 0.0

            avg_prob = np.mean(probs)

            # 如果平均值 < 0，说明曲率增强有帮助
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

    def val_robustness_link(self, edges_index, edges_weight, select_edges, num_add, num_remove,
                            subadj, num_val, all_edges, sub_features, output_goal, pos_edge_index):
        """链接预测的鲁棒性评估（与 deeplift_ricci_train.py 中的 val_robustness 一致）"""
        result_dict = dict()
        result_dict_KL = dict()

        predict_old_label = np.argmax(softmax(output_goal))

        edges_old_dict = dict()
        for i in range(len(edges_index[0])):
            edges_old_dict[str(edges_index[0][i].item()) + ',' + str(edges_index[1][i].item())] = i

        edges_old_dict_reverse = dict()
        for key, value in edges_old_dict.items():
            node_list = key.split(',')
            edges_old_dict_reverse[value] = [int(node_list[0]), int(node_list[1])]

        for count in range(num_val):
            tmp_changed_edgelist = []

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

            # 评估扰动后的预测
            tmp_output_encode = self.model.encode(
                sub_features,
                torch.tensor(edges_index_new),
                torch.tensor(edges_index_new),
                edge_weight1=torch.tensor(edges_weight_new, dtype=torch.float32),
                edge_weight2=torch.tensor(edges_weight_new, dtype=torch.float32)
            )
            tmp_output = self.model.decode(tmp_output_encode, pos_edge_index).view(-1)

            result_dict[count] = float(abs(
                softmax(tmp_output.detach().numpy())[predict_old_label] -
                softmax(output_goal)[predict_old_label]
            ))

            result_dict_KL[count] = KL_divergence(
                softmax(output_goal),
                softmax(tmp_output.detach().numpy())
            )

        return result_dict, result_dict_KL

    def ave(self, my_dict):
        """计算字典值的平均"""
        non_zero_values = [v for v in my_dict.values() if v != float('inf')]
        if non_zero_values:
            return sum(non_zero_values) / len(non_zero_values), len(non_zero_values)
        return 0, 0

    def evaluate_edge_with_lambda(self, target_edge: list, lam: float) -> Optional[float]:
        """
        评估边在给定 lambda 下的鲁棒性差值
        返回: prob_robust(curvature) - prob_robust(base)
        如果曲率增强后更鲁棒，返回负值
        """
        # 加载子图数据
        sub_features, sub_old, map_edge_index_old, sub_graph_old, \
            map_edge_weight_old, map_edge_old_dict, submapping, map_edge_old_dict_reverse = \
            self.data_loader.gen_new_edge(
                target_edge, self.model, self.edges_old, self.clear_time_dict, self.features, self.cfg.normalize_adj
            )

        goal_1 = submapping[target_edge[0]]
        goal_2 = submapping[target_edge[1]]

        pos_edge_index = [[goal_1], [goal_2]]
        pos_edge_index = torch.tensor(pos_edge_index)

        # 模型前向传播
        encode_logits_old = self.model.encode(
            sub_features, map_edge_index_old, map_edge_index_old,
            map_edge_weight_old, map_edge_weight_old
        )
        decode_logits_old = self.model.decode(encode_logits_old, pos_edge_index).view(-1)
        decode_logits_old_numpy = decode_logits_old.detach().numpy()

        predict_old_label = np.argmax(softmax(decode_logits_old_numpy))

        # 获取路径
        path_ceshi_goal1 = dfs2(goal_1, goal_1, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_1 = reverse_paths(path_ceshi_goal1)

        path_ceshi_goal2 = dfs2(goal_2, goal_2, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_2 = reverse_paths(path_ceshi_goal2)

        # 提取目标边列表
        target1_changed_edgelist = []
        for path in target_path_1:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[1], path[2]])

        target1_changed_edgelist = clear(target1_changed_edgelist)

        target2_changed_edgelist = []
        for path in target_path_2:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[1], path[2]])

        target2_changed_edgelist = clear(target2_changed_edgelist)

        # 合并目标边
        target_changed_edgelist = []
        for edge in target1_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)
        for edge in target2_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)

        # 检查边数量
        num_remove_val = math.floor(
            self.cfg.num_remove_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))
        num_add_val = math.floor(
            self.cfg.num_add_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))

        if len(target_changed_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            return 0

        # 构建 target_edgelist_index 和 weight
        all_edges_list = []
        for edge in target1_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)
        for edge in target2_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)

        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in all_edges_list:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])
                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(sub_old[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight, dtype=torch.float32)

        # 获取 edge_mask
        edge_mask = self.explain_edge(target_edge, use_cache=True)
        if edge_mask is None:
            return 0

        # 计算曲率
        curvature_result = self.compute_curvature(map_edge_index_old, map_edge_weight_old, sub_old)

        # 计算选择的边数量
        select_edge_important = math.ceil(
            self.cfg.sparsity * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))

        # ========== 基础选择（不使用曲率）==========
        if self.cfg.method == "convex":
            # Convex 方法: edge_mask 对应 target_changed_edgelist
            # edge_mask 是 list 形式，长度等于 target_changed_edgelist
            total_result_base = dict()
            for i, edge in enumerate(target_changed_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                if i < len(edge_mask):
                    total_result_base[edge_str] = edge_mask[i] if isinstance(edge_mask[i], (int, float)) else edge_mask[
                        i].item()
        else:
            # DeepLIFT, FlowX, GNN-LRP: edge_mask 对应 map_edge_index_old
            total_result_base = dict()
            for i in range(len(map_edge_index_old[0])):
                edge_str = str(map_edge_index_old[0][i].item()) + ',' + str(map_edge_index_old[1][i].item())
                total_result_base[edge_str] = edge_mask[i].item()

        sort_diff_base = sorted(total_result_base.items(), key=lambda item: item[1], reverse=True)

        count_number = 0
        select_edges_list_old_base = []
        i = 0
        while count_number < select_edge_important and i < len(sort_diff_base):
            tmp = []
            s1 = sort_diff_base[i][0].split(',')
            for j in s1:
                tmp.append(int(j))
            if [tmp[0], tmp[1]] not in select_edges_list_old_base and [tmp[1],
                                                                       tmp[0]] not in select_edges_list_old_base:
                select_edges_list_old_base.append(tmp)
                count_number += 1
            i += 1

        # ========== 曲率增强选择 ==========
        if self.cfg.method == "convex":
            total_result_curvature = dict()
            for i, edge in enumerate(target_changed_edgelist):
                edge_str = str(edge[0]) + ',' + str(edge[1])
                curvature_val = curvature_result.get(edge_str, 0)
                if i < len(edge_mask):
                    mask_val = edge_mask[i] if isinstance(edge_mask[i], (int, float)) else edge_mask[i].item()
                    total_result_curvature[edge_str] = self.compute_combined_score(mask_val, curvature_val, lam)
        else:
            total_result_curvature = dict()
            for i in range(len(map_edge_index_old[0])):
                edge_str = str(map_edge_index_old[0][i].item()) + ',' + str(map_edge_index_old[1][i].item())
                curvature_val = curvature_result.get(edge_str, 0)
                total_result_curvature[edge_str] = self.compute_combined_score(edge_mask[i].item(), curvature_val, lam)

        sort_diff_curvature = sorted(total_result_curvature.items(), key=lambda item: item[1], reverse=True)

        count_number = 0
        select_edges_list_old = []
        i = 0
        while count_number < select_edge_important and i < len(sort_diff_curvature):
            tmp = []
            s1 = sort_diff_curvature[i][0].split(',')
            for j in s1:
                tmp.append(int(j))
            if [tmp[0], tmp[1]] not in select_edges_list_old and [tmp[1], tmp[0]] not in select_edges_list_old:
                select_edges_list_old.append(tmp)
                count_number += 1
            i += 1

        # 检查选择是否不同
        select_flag = True
        for edge in select_edges_list_old:
            if edge not in select_edges_list_old_base and [edge[1], edge[0]] not in select_edges_list_old_base:
                select_flag = False
                break

        if select_flag:
            # 选择相同，返回 0
            return 0

        # ========== 评估鲁棒性 ==========
        num_val = 100

        # 曲率增强后的鲁棒性
        result_val_prob, result_val_kl = self.val_robustness_link(
            target_edgelist_index, target_edgelist_weight, select_edges_list_old,
            num_add_val, num_remove_val, sub_old, num_val, len(target_changed_edgelist),
            sub_features, decode_logits_old_numpy, pos_edge_index
        )
        prob_robust, _ = self.ave(result_val_prob)

        # 基础选择的鲁棒性
        result_val_prob_base, result_val_kl_base = self.val_robustness_link(
            target_edgelist_index, target_edgelist_weight, select_edges_list_old_base,
            num_add_val, num_remove_val, sub_old, num_val, len(target_changed_edgelist),
            sub_features, decode_logits_old_numpy, pos_edge_index
        )
        prob_robust_base, _ = self.ave(result_val_prob_base)

        # 返回差值
        return prob_robust - prob_robust_base

    def evaluate_val_edge(self, target_edge: list, good_lambdas_list: list):
        """评估验证边在多个 lambda 下的鲁棒性，并保存结果"""

        # 加载子图数据
        sub_features, sub_old, map_edge_index_old, sub_graph_old, \
            map_edge_weight_old, map_edge_old_dict, submapping, map_edge_old_dict_reverse = \
            self.data_loader.gen_new_edge(
                target_edge, self.model, self.edges_old, self.clear_time_dict, self.features, self.cfg.normalize_adj
            )

        goal_1 = submapping[target_edge[0]]
        goal_2 = submapping[target_edge[1]]

        pos_edge_index = [[goal_1], [goal_2]]
        pos_edge_index = torch.tensor(pos_edge_index)

        # 模型前向传播
        encode_logits_old = self.model.encode(
            sub_features, map_edge_index_old, map_edge_index_old,
            map_edge_weight_old, map_edge_weight_old
        )
        decode_logits_old = self.model.decode(encode_logits_old, pos_edge_index).view(-1)
        decode_logits_old_numpy = decode_logits_old.detach().numpy()

        predict_old_label = np.argmax(softmax(decode_logits_old_numpy))

        # 获取路径
        path_ceshi_goal1 = dfs2(goal_1, goal_1, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_1 = reverse_paths(path_ceshi_goal1)

        path_ceshi_goal2 = dfs2(goal_2, goal_2, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_2 = reverse_paths(path_ceshi_goal2)

        # 提取目标边列表
        target1_changed_edgelist = []
        for path in target_path_1:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[1], path[2]])

        target1_changed_edgelist = clear(target1_changed_edgelist)

        target2_changed_edgelist = []
        for path in target_path_2:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[1], path[2]])

        target2_changed_edgelist = clear(target2_changed_edgelist)

        # 合并目标边
        target_changed_edgelist = []
        for edge in target1_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)
        for edge in target2_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)

        # 检查边数量
        num_remove_val = math.floor(
            self.cfg.num_remove_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))
        num_add_val = math.floor(
            self.cfg.num_add_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))

        if len(target_changed_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Edge {target_edge}: Skipped (edges={len(target_changed_edgelist)})")
            return

        # 构建 target_edgelist_index 和 weight
        all_edges_list = []
        for edge in target1_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)
        for edge in target2_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)

        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in all_edges_list:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])
                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(sub_old[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight, dtype=torch.float32)

        # 计算曲率
        curvature_result = self.compute_curvature(map_edge_index_old, map_edge_weight_old, sub_old)

        # 获取 edge_mask
        edge_mask = self.explain_edge(target_edge, use_cache=True)
        if edge_mask is None:
            return

        # 边选择数量
        select_edge_important = math.ceil(
            self.cfg.sparsity * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))

        num_val = 100

        # 随机选择边的鲁棒性（作为基准）
        random_select_edges = random.sample(target_changed_edgelist,
                                            min(select_edge_important, len(target_changed_edgelist)))
        withoutcon_result_val_prob, withoutcon_result_val_kl = self.val_robustness_link(
            target_edgelist_index, target_edgelist_weight, random_select_edges,
            num_add_val, num_remove_val, sub_old, num_val, len(target_changed_edgelist),
            sub_features, decode_logits_old_numpy, pos_edge_index
        )

        # 对每个 lambda 进行评估
        for lam in good_lambdas_list:
            result_important_dict = dict()
            result_important_base_dict = dict()
            save_flag = False

            # ========== 基础选择（不使用曲率）==========
            if self.cfg.method == "convex":
                total_result_base = dict()
                for i, edge in enumerate(target_changed_edgelist):
                    edge_str = str(edge[0]) + ',' + str(edge[1])
                    if i < len(edge_mask):
                        total_result_base[edge_str] = edge_mask[i] if isinstance(edge_mask[i], (int, float)) else \
                        edge_mask[i].item()
            else:
                total_result_base = dict()
                for i in range(len(map_edge_index_old[0])):
                    edge_str = str(map_edge_index_old[0][i].item()) + ',' + str(map_edge_index_old[1][i].item())
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

            # ========== 曲率增强选择 ==========
            if self.cfg.method == "convex":
                total_result_curvature = dict()
                for i, edge in enumerate(target_changed_edgelist):
                    edge_str = str(edge[0]) + ',' + str(edge[1])
                    curvature_val = curvature_result.get(edge_str, 0)
                    if i < len(edge_mask):
                        mask_val = edge_mask[i] if isinstance(edge_mask[i], (int, float)) else edge_mask[i].item()
                        total_result_curvature[edge_str] = self.compute_combined_score(mask_val, curvature_val, lam)
            else:
                total_result_curvature = dict()
                for i in range(len(map_edge_index_old[0])):
                    edge_str = str(map_edge_index_old[0][i].item()) + ',' + str(map_edge_index_old[1][i].item())
                    curvature_val = curvature_result.get(edge_str, 0)
                    total_result_curvature[edge_str] = self.compute_combined_score(edge_mask[i].item(), curvature_val,
                                                                                   lam)

            sort_diff_curvature = sorted(total_result_curvature.items(), key=lambda x: x[1], reverse=True)

            select_gnn_edgelist = []
            select_idx = 0
            while len(select_gnn_edgelist) < select_edge_important and select_idx < len(sort_diff_curvature):
                edge_str = sort_diff_curvature[select_idx][0]
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

                # 评估曲率增强的选择
                result_val_prob, result_val_kl = self.val_robustness_link(
                    target_edgelist_index, target_edgelist_weight, select_gnn_edgelist,
                    num_add_val, num_remove_val, sub_old, num_val, len(target_changed_edgelist),
                    sub_features, decode_logits_old_numpy, pos_edge_index
                )

                idx_important = 0
                result_important_dict[f'{idx_important},select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])] for e
                                                                                           in select_gnn_edgelist]
                result_important_dict[f'{idx_important},robustness {self.cfg.method} prob'] = result_val_prob
                result_important_dict[f'{idx_important},robustness {self.cfg.method} kl'] = result_val_kl
                result_important_dict[
                    f'{idx_important},robustness {self.cfg.method} prob withoutcon'] = withoutcon_result_val_prob
                result_important_dict[
                    f'{idx_important},robustness {self.cfg.method} kl withoutcon'] = withoutcon_result_val_kl

                # 评估基础选择
                result_val_prob_base, result_val_kl_base = self.val_robustness_link(
                    target_edgelist_index, target_edgelist_weight, select_gnn_base_edgelist,
                    num_add_val, num_remove_val, sub_old, num_val, len(target_changed_edgelist),
                    sub_features, decode_logits_old_numpy, pos_edge_index
                )

                result_important_base_dict[f'{idx_important},select {self.cfg.method} edge'] = [[int(e[0]), int(e[1])]
                                                                                                for e in
                                                                                                select_gnn_base_edgelist]
                result_important_base_dict[f'{idx_important},robustness {self.cfg.method} prob'] = result_val_prob_base
                result_important_base_dict[f'{idx_important},robustness {self.cfg.method} kl'] = result_val_kl_base
                result_important_base_dict[
                    f'{idx_important},robustness {self.cfg.method} prob withoutcon'] = withoutcon_result_val_prob
                result_important_base_dict[
                    f'{idx_important},robustness {self.cfg.method} kl withoutcon'] = withoutcon_result_val_kl

            # 保存结果
            if save_flag:
                save_dir = self.get_lambda_save_dir()
                lam_save_dir = os.path.join(save_dir, str(lam), self.cfg.curvature_type)
                os.makedirs(lam_save_dir, exist_ok=True)

                edge_str = f"{target_edge[0]}_{target_edge[1]}"
                with open(os.path.join(lam_save_dir, f"{edge_str}.json"), 'w') as f:
                    json.dump(result_important_dict, f)
                with open(os.path.join(lam_save_dir, f"{edge_str}_base.json"), 'w') as f:
                    json.dump(result_important_base_dict, f)

                print(f"Edge {target_edge}: Saved results for lambda={lam}")


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
        self.data_loader = gen_link_data(
            self.cfg.dataset,
            os.path.join(self.cfg.data_root, self.cfg.dataset),
            self.cfg.start_time,
            self.cfg.end_time,
            self.cfg.time_flag,
            self.cfg.layer_numbers
        )

        self.dataset, self.edges_old, self.clear_time_dict = self.data_loader.load_data()
        self.data = self.dataset[0]

        self.features = self.data.x.to(torch.float32)
        self.num_nodes = self.data.num_nodes
        self.modelname = self.cfg.dataset

        # 构建目标边列表（去除自环和重复）
        self.target_edges_list = []
        for i in range(len(self.edges_old[0])):
            node_1 = self.edges_old[0][i].item()
            node_2 = self.edges_old[1][i].item()
            if node_1 != node_2 and [node_1, node_2] not in self.target_edges_list and [node_2,
                                                                                        node_1] not in self.target_edges_list:
                self.target_edges_list.append([node_1, node_2])

        print(f"Loaded dataset: {self.cfg.dataset}")
        print(f"  Nodes: {self.num_nodes}, Edges: {len(self.target_edges_list)}")

    def load_models(self):
        """加载预训练的链接预测模型"""
        # 确定模型路径（参考 train_GCN.py 的保存格式）
        norm_flag = "norm" if self.cfg.normalize_adj else "nonorm"
        model_path = f'./checkpoints/link_{self.modelname}_{norm_flag}.pt'

        print(f'model_path: {model_path}')

        # 创建模型（与 train_GCN.py 中的 NetLinkTrain 结构一致）
        from torch_geometric.nn import GCNConv

        class NetLinkEval(torch.nn.Module):
            def __init__(self, nfeat, nhid):
                super(NetLinkEval, self).__init__()
                self.conv1 = GCNConv(nfeat, nhid, add_self_loops=False, normalize=False, bias=False)
                self.conv2 = GCNConv(nhid, nhid, add_self_loops=False, normalize=False, bias=False)
                self.linear = nn.Linear(nhid * 2, 2, bias=False)

            def encode(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2):
                """与原始 Net_link.encode 一致的签名"""
                x = self.conv1(x.to(torch.float32), edge_index1, edge_weight=edge_weight1)
                x = x.relu()
                return self.conv2(x, edge_index2, edge_weight=edge_weight2)

            def decode(self, z, pos_edge_index):
                edge_index = pos_edge_index
                h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
                h = self.linear(h)
                return h

            def forward(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2, pos_edge_index):
                z = self.encode(x, edge_index1, edge_index2, edge_weight1, edge_weight2)
                z = self.decode(z, pos_edge_index)
                return z

            def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
                x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
                x_1 = F.relu(x_0)
                return (x_0, x_1)

        # 创建模型实例
        self.model = NetLinkEval(
            nfeat=self.data.num_features,
            nhid=self.cfg.hidden_dim
        ).to(self.cfg.device)

        self.model.eval()

        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            # train_GCN.py 保存格式: {"task": task, "dataset": dataset, "state": model.state_dict()}
            self.model.load_state_dict(checkpoint['state'])
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
            print("Please train a model first using: python train_GCN.py --task link_prediction --dataset UCI")

        # 提取权重矩阵（用于 convex 方法）
        self.W1 = self.model.state_dict()['conv1.lin.weight'].t()
        self.W2 = self.model.state_dict()['conv2.lin.weight'].t()
        self.W3 = self.model.state_dict()['linear.weight'].t()

        # 创建用于评估的模型（与主模型共享权重）

        if self.cfg.method in ["gnnexplainer", "pgexplainer"]:
            self.evaulate_model = NetLinkEvaluatePYG(
                nfeat=self.data.num_features,
                nhid=self.cfg.hidden_dim
            ).to(self.cfg.device)


        else:
            self.evaulate_model = NetLinkEvaluate(
                nfeat=self.data.num_features,
                nhid=self.cfg.hidden_dim
            ).to(self.cfg.device)

        eval_model_dict = self.evaulate_model.state_dict()
        if self.cfg.method in ["gnnexplainer", "pgexplainer"]:
            # 标准 PyG GCNConv 使用 conv1.lin.weight
            eval_model_dict['conv1.lin.weight'] = self.model.state_dict()['conv1.lin.weight']
            eval_model_dict['conv2.lin.weight'] = self.model.state_dict()['conv2.lin.weight']
            eval_model_dict['linear.weight'] = self.model.state_dict()['linear.weight']
        else:
            # GCNConvExplainer 使用 conv1.weight，需要转置
            eval_model_dict['conv1.weight'] = self.model.state_dict()['conv1.lin.weight'].t()
            eval_model_dict['conv2.weight'] = self.model.state_dict()['conv2.lin.weight'].t()
            eval_model_dict['linear.weight'] = self.model.state_dict()['linear.weight']

            # self.evaulate_model.eval()
            # # 复制权重到评估模型
            # eval_model_dict = self.evaulate_model.state_dict()
            # eval_model_dict['conv1.weight'] = self.model.state_dict()['conv1.lin.weight'].t()
            # eval_model_dict['conv2.weight'] = self.model.state_dict()['conv2.lin.weight'].t()
            # eval_model_dict['linear.weight'] = self.model.state_dict()['linear.weight']
        self.evaulate_model.load_state_dict(eval_model_dict)





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

    def explain_edge(self, target_edge: list, use_cache: bool = True) -> Optional[torch.Tensor]:
        """对单个边进行解释"""
        sub_features, sub_old, map_edge_index_old, sub_graph_old, \
            map_edge_weight_old, map_edge_old_dict, submapping, map_edge_old_dict_reverse = \
            self.data_loader.gen_new_edge(
                target_edge, self.model, self.edges_old, self.clear_time_dict, self.features,self.cfg.normalize_adj
            )
        self.model.eval()

        goal_1 = submapping[target_edge[0]]
        goal_2 = submapping[target_edge[1]]

        pos_edge_index = [[goal_1], [goal_2]]
        pos_edge_index = torch.tensor(pos_edge_index)

        # 模型前向传播获取预测标签
        encode_logits_old = self.model.encode(
            sub_features, map_edge_index_old, map_edge_index_old,
            map_edge_weight_old, map_edge_weight_old
        )
        decode_logits_old = self.model.decode(encode_logits_old, pos_edge_index).view(-1)
        decode_logits_old_numpy = decode_logits_old.detach().numpy()

        G_old = softmax(decode_logits_old_numpy)
        predict_old_label = np.argmax(G_old)

        # 获取两个端点的路径（DFS）
        path_ceshi_goal1 = dfs2(goal_1, goal_1, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_1 = reverse_paths(path_ceshi_goal1)

        path_ceshi_goal2 = dfs2(goal_2, goal_2, sub_graph_old, self.cfg.layer_numbers + 1, [], [])
        target_path_2 = reverse_paths(path_ceshi_goal2)

        # 提取 goal_1 的目标边列表
        target1_changed_edgelist = []
        for path in target_path_1:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target1_changed_edgelist.append([path[1], path[2]])

        target1_changed_edgelist = clear(target1_changed_edgelist)

        # 提取 goal_2 的目标边列表
        target2_changed_edgelist = []
        for path in target_path_2:
            key_1 = str(path[0]) + ',' + str(path[1])
            key_2 = str(path[1]) + ',' + str(path[0])
            key_3 = str(path[1]) + ',' + str(path[2])
            key_4 = str(path[2]) + ',' + str(path[1])

            if key_1 in map_edge_old_dict.keys() or key_2 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[0], path[1]])
            if key_3 in map_edge_old_dict.keys() or key_4 in map_edge_old_dict.keys():
                target2_changed_edgelist.append([path[1], path[2]])

        target2_changed_edgelist = clear(target2_changed_edgelist)

        # 合并所有边
        all_edges_list = []
        for edge in target1_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)

        for edge in target2_changed_edgelist:
            if [edge[0], edge[1]] not in all_edges_list and [edge[1], edge[0]] not in all_edges_list:
                all_edges_list.append(edge)

        # 构建 target_edgelist_index 和 target_edgelist_weight
        target_edgelist_index = [[], []]
        target_edgelist_weight = []
        for edge in all_edges_list:
            if edge[0] != edge[1]:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])

                target_edgelist_index[0].append(edge[1])
                target_edgelist_index[1].append(edge[0])
                target_edgelist_weight.append(sub_old[edge[1], edge[0]])
            else:
                target_edgelist_index[0].append(edge[0])
                target_edgelist_index[1].append(edge[1])
                target_edgelist_weight.append(sub_old[edge[0], edge[1]])

        target_edgelist_index = torch.LongTensor(target_edgelist_index)
        target_edgelist_weight = torch.tensor(target_edgelist_weight)
        target_edgelist_weight = target_edgelist_weight.to(torch.float32)

        # 合并目标边列表
        target_changed_edgelist = []
        for edge in target1_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)
        for edge in target2_changed_edgelist:
            if edge not in target_changed_edgelist:
                target_changed_edgelist.append(edge)

        # 检查边数量是否足够
        num_remove_val = math.floor(
            self.cfg.num_remove_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))
        num_add_val = math.floor(
            self.cfg.num_add_val_ratio * (len(target1_changed_edgelist) + len(target2_changed_edgelist)))

        if len(target_changed_edgelist) < 20 or num_remove_val <= 0 or num_add_val <= 0:
            print(f"Edge {target_edge}: Skipped (edges={len(target_changed_edgelist)})")
            return None

        # 缓存路径
        save_dir = self.get_save_dir()
        edge_str = f"{target_edge[0]}_{target_edge[1]}"
        if self.cfg.method == "convex":
            cache_path = os.path.join(save_dir, f"{self.cfg.method}_{edge_str}.json")
        else:
            cache_path = os.path.join(save_dir, f"{self.cfg.method}_{edge_str}.npy")

        if use_cache and os.path.exists(cache_path):
            if self.cfg.method == "convex":
                json_save_path = os.path.join(save_dir, f"{self.cfg.method}_{edge_str}.json")
                if os.path.exists(json_save_path):
                    with open(json_save_path, 'r') as f:
                        select_edges_list_value = json.load(f)
                    edge_mask = torch.tensor(select_edges_list_value, dtype=torch.float32)
                    print(f"Edge {target_edge}: Loaded from cache")
                    return edge_mask
            else:
                npy_save_path = os.path.join(save_dir, f"{self.cfg.method}_{edge_str}.npy")
                if os.path.exists(npy_save_path):
                    edge_mask_array = np.load(npy_save_path)
                    edge_mask = torch.tensor(edge_mask_array, dtype=torch.float32)
                    print(f"Edge {target_edge}: Loaded from cache")
                    return edge_mask
        else:
            if self.cfg.method == "deeplift":
                # print('deeplift',deeplift)
                from baselines.dig.xgraph.method import DeepLIFT_link
                explainer = DeepLIFT_link(self.evaulate_model, explain_graph=True)
                sparsity = 0

                masks_old, _, _ = explainer(
                    sub_features, map_edge_index_old, sparsity=sparsity,
                    given_class=predict_old_label,
                    edge_weight=map_edge_weight_old, pos_edge_index=pos_edge_index
                )

                edge_mask = masks_old[0]

            elif self.cfg.method == "flowx":
                from baselines.dig.xgraph.method import FlowMask_link

                explainer = FlowMask_link(self.evaulate_model, explain_graph=True)
                sparsity = 0


                _, masks_old, _ = explainer(sub_features, map_edge_index_old, sparsity=sparsity,
                                            given_class=predict_old_label,
                                            edge_weight=map_edge_weight_old, pos_edge_index=pos_edge_index)

                edge_mask = masks_old[0]

            elif self.cfg.method == "gnnlrp":
                from baselines.dig.xgraph.method import GNN_LRP_link
                explainer = GNN_LRP_link(self.evaulate_model, explain_graph=True)
                sparsity = 0

                _, masks_old, _ = explainer(sub_features, map_edge_index_old, sparsity=sparsity,
                                            given_class=predict_old_label,
                                            edge_weight=map_edge_weight_old, pos_edge_index=pos_edge_index,
                                            goal_1=goal_1,
                                            goal_2=goal_2)

                edge_mask = masks_old[0]

                edge_mask = custom_log_transform(edge_mask)
            elif self.cfg.method == "convex":
                relu_delta, relu_end, relu_start = self.data_loader.gen_parameters_v2(
                    self.model, sub_features, map_edge_index_old, map_edge_weight_old
                )

                _, _, test_edge_result_1 = test_path_contribution_edge(
                    target_path_1,
                    csr_matrix(sub_old.shape),
                    sub_old,
                    target1_changed_edgelist,
                    relu_delta,
                    relu_start,
                    relu_end,
                    sub_features,
                    self.W1,
                    self.W2
                )
                target1_edge_result = map_target(test_edge_result_1, goal_1)

                # 计算 goal_2 的边贡献
                _, _, test_edge_result_2 = test_path_contribution_edge(
                    target_path_2,
                    csr_matrix(sub_old.shape),
                    sub_old,
                    target2_changed_edgelist,
                    relu_delta,
                    relu_start,
                    relu_end,
                    sub_features,
                    self.W1,
                    self.W2
                )
                target2_edge_result = map_target(test_edge_result_2, goal_2)

                # 应用 MLP 贡献（W3 分为两半，对应两个节点的 embedding 拼接）
                final_target1_edge_result = mlp_contribution(
                    target1_edge_result,
                    self.W3[:encode_logits_old.shape[1]].detach().numpy()
                )
                final_target2_edge_result = mlp_contribution(
                    target2_edge_result,
                    self.W3[encode_logits_old.shape[1]:].detach().numpy()
                )

                # 合并两个目标节点的边贡献
                target_edge_result = dict()
                for key, value in final_target1_edge_result.items():
                    if key not in target_edge_result.keys():
                        target_edge_result[key] = value
                    else:
                        target_edge_result[key] += value

                for key, value in final_target2_edge_result.items():
                    if key not in target_edge_result.keys():
                        target_edge_result[key] = value
                    else:
                        target_edge_result[key] += value

                # 计算选择的边数量
                select_edge_important = math.ceil(
                    self.cfg.sparsity * (len(target1_changed_edgelist) + len(target2_changed_edgelist))
                )

                # 使用凸优化计算边的重要性分数
                select_edges_list_value, select_edges_list_sort = main_con_edge(
                    select_edge_important,
                    target_edge_result,
                    target_changed_edgelist,
                    [0, 0],  # old_tensor (初始化为 0)
                    decode_logits_old_numpy
                )

                edge_mask = torch.tensor(select_edges_list_value, dtype=torch.float32)

            elif self.cfg.method == "gnnexplainer":
                from baselines.torch_geometric.explain import GNNExplainer, Explainer

                explainer = Explainer(
                    model=self.evaulate_model,
                    explanation_type='phenomenon',
                    algorithm=GNNExplainer(epochs=200, lr=0.0001),
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='edge',
                        return_type='raw'
                    )
                )
                explanation_old = explainer(
                    x=sub_features,
                    edge_index=map_edge_index_old,
                    target=torch.tensor([predict_old_label]),
                    edge_weight=map_edge_weight_old,
                    pos_edge_index=pos_edge_index
                )
                edge_mask = explanation_old.edge_mask

            elif self.cfg.method == "pgexplainer":
                from baselines.torch_geometric.explain import PGExplainer, Explainer

                train_epoches = 100
                train_lr = 0.001

                explainer_old = Explainer(
                    model=self.evaulate_model,
                    explanation_type='phenomenon',
                    algorithm=PGExplainer(epochs=train_epoches, lr=train_lr),
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='graph',
                        return_type='raw'
                    )
                )

                for epoch in range(train_epoches):
                    loss_old = explainer_old.algorithm.train(epoch, self.evaulate_model, sub_features, map_edge_index_old,
                                                             target=torch.tensor([predict_old_label]),
                                                             edge_weight=map_edge_weight_old,
                                                             pos_edge_index=pos_edge_index)
                explanation_old = explainer_old(sub_features, map_edge_index_old,
                                                target=torch.tensor([predict_old_label]),
                                                edge_weight=map_edge_weight_old, pos_edge_index=pos_edge_index)
                edge_mask = explanation_old.edge_mask






            # 后处理（如果是 GNN-LRP 则进行 log 变换）



            if self.cfg.method == "convex":
                json_save_path = os.path.join(save_dir, f"{self.cfg.method}_{edge_str}.json")
                with open(json_save_path, 'w') as f:
                    json.dump(select_edges_list_value, f)
                print(f"target edge {target_edge}: Computed and saved")
            else:
                np.save(cache_path, edge_mask.detach().cpu().numpy())
                print(f"target edge {target_edge}: Computed and saved")


        return edge_mask






# ========== 命令行解析 ==========
def parse_args() -> ExplainConfig:
    p = argparse.ArgumentParser(description="Unified Link Explanation with Curvature")

    # 数据集
    p.add_argument("--dataset", type=str, default="bitcoinalpha",
                   choices=["UCI", "bitcoinalpha", "bitcoinotc"])
    p.add_argument("--data_root", type=str, default="./data")

    # 时间窗口
    p.add_argument("--start_time", type=int, default=0)
    p.add_argument("--end_time", type=int, default=20)
    p.add_argument("--time_flag", type=str, default="week")

    # 模型
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layer_numbers", type=int, default=2)

    # 解释方法
    p.add_argument("--method", type=str, default="gnnexplainer",
                   choices=["deeplift", "flowx", "gnnexplainer", "gnnlrp", "pgexplainer", "convex"])
    p.add_argument("--sparsity", type=float, default=0.1)

    # Curvature
    p.add_argument("--curvature_type", type=str, default="ricci",
                   choices=["ricci", "resistance"])
    p.add_argument("--ricci_alpha", type=float, default=0.0)
    p.add_argument("--resistance_epsilon", type=float, default=0.01)
    p.add_argument("--resistance_method", type=str, default="kts")

    # 采样
    p.add_argument("--sample_num_edges", type=int, default=100)

    p.add_argument("--max_train_edges", type=int, default=250)

    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--percentage_threshold", type=float, default=0.9)

    # 扰动
    p.add_argument("--num_remove_val_ratio", type=float, default=0.1)
    p.add_argument("--num_add_val_ratio", type=float, default=0.1)

    # 运行模式
    p.add_argument("--run_mode", type=str, default="val",
                   choices=["train", "val", "select"])
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.1)

    # 其他
    p.add_argument("--result_root", type=str, default="./result")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize_adj",  default=False)

    args = p.parse_args()

    # 根据数据集自动设置 time_flag
    if args.dataset == "UCI":
        time_flag = "week"
    else:
        time_flag = "month"

    return ExplainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        start_time=args.start_time,
        end_time=args.end_time,
        time_flag=time_flag,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        layer_numbers=args.layer_numbers,
        method=args.method,
        sparsity=args.sparsity,
        curvature_type=args.curvature_type,
        ricci_alpha=args.ricci_alpha,
        resistance_epsilon=args.resistance_epsilon,
        resistance_method=args.resistance_method,
        sample_num_edges=args.sample_num_edges,
        train_ratio=args.train_ratio,
        percentage_threshold=args.percentage_threshold,
        num_remove_val_ratio=args.num_remove_val_ratio,
        num_add_val_ratio=args.num_add_val_ratio,
        run_mode=args.run_mode,
        n_trials=args.n_trials,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        result_root=args.result_root,
        seed=args.seed,
        normalize_adj=args.normalize_adj,
    )


# ========== 主函数 ==========
def main():
    cfg = parse_args()
    print(f"Config: {cfg}")

    explainer = LinkExplainer(cfg)

    # 运行批量解释
    results = explainer.run()

    # 保存结果统计



if __name__ == "__main__":
    main()