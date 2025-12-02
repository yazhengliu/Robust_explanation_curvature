

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from dig.xgraph.method import DeepLIFT
from dig.xgraph.method import FlowMask
from torch_geometric.explain import GNNExplainer, Explainer
from dig.xgraph.method import GNN_LRP

from utils.node_utils import test_path_contribution_edge, map_target, main_con_edge
from scipy.sparse import csr_matrix
# 解释方法枚举
class ExplainerMethod(Enum):
    DEEPLIFT = "deeplift"
    FLOWX = "flowx"
    GNNEXPLAINER = "gnnexplainer"
    GNNLRP = "gnnlrp"
    PGEXPLAINER = "pgexplainer"
    CONVEX = "convex"  # 你的自定义方法


# 抽象基类
class BaseExplainer(ABC):
    """GNN解释器的抽象基类"""

    def __init__(self, model: nn.Module, explain_graph: bool = False):
        self.model = model
        self.explain_graph = explain_graph
        self.model.eval()

    @abstractmethod
    def explain(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            node_idx: int,
            edge_weight: Optional[torch.Tensor] = None,
            num_classes: Optional[int] = None,
            target_class: Optional[int] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        生成边的解释掩码

        Args:
            x: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            node_idx: 目标节点索引
            edge_weight: 边权重 [E]
            num_classes: 类别数
            target_class: 目标类别（预测标签）

        Returns:
            edge_mask: 边的重要性分数 [E]
        """
        pass

    def get_predicted_label(self, x: torch.Tensor, edge_index: torch.Tensor,
                            edge_weight: torch.Tensor, node_idx: int) -> int:
        """获取模型预测标签"""
        with torch.no_grad():
            output = self.model(x, edge_index, edge_weight)
            probs = F.softmax(output[node_idx], dim=0).numpy()
            return int(np.argmax(probs))


# DeepLIFT 解释器
class DeepLIFTExplainer(BaseExplainer):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, explain_graph=False)

        self.explainer = DeepLIFT(model, explain_graph=False)

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None, **kwargs) -> torch.Tensor:
        if target_class is None:
            target_class = self.get_predicted_label(x, edge_index, edge_weight, node_idx)

        sparsity = kwargs.get('sparsity', 1)
        _, masks, _ = self.explainer(x, edge_index, sparsity=sparsity,
                                     num_classes=num_classes, node_idx=node_idx,
                                     edge_weight=edge_weight)
        edge_mask = masks[target_class]
        return edge_mask


# FlowX 解释器
class FlowXExplainer(BaseExplainer):
    def __init__(self, model: nn.Module, lr: float = 0.001, epochs: int = 200, **kwargs):
        super().__init__(model, explain_graph=False)

        self.explainer = FlowMask(model, explain_graph=False, lr=lr, epochs=epochs)

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None, **kwargs) -> torch.Tensor:
        if target_class is None:
            target_class = self.get_predicted_label(x, edge_index, edge_weight, node_idx)

        sparsity = kwargs.get('sparsity', 0)
        _, masks, _ = self.explainer(x, edge_index, sparsity=sparsity,
                                     num_classes=num_classes, node_idx=node_idx,
                                     edge_weight=edge_weight)
        edge_mask = masks[target_class]
        return edge_mask


# GNNExplainer 解释器
class GNNExplainerWrapper(BaseExplainer):
    def __init__(self, model: nn.Module, epochs: int = 200, lr: float = 0.001, **kwargs):
        super().__init__(model, explain_graph=False)

        self.epochs = epochs
        self.lr = lr

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None, **kwargs) -> torch.Tensor:
        from torch_geometric.explain import GNNExplainer, Explainer

        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw'
            ),
        )

        explanation = explainer(x, edge_index, index=node_idx, edge_weight=edge_weight)
        return explanation.edge_mask


# GNN-LRP 解释器
class GNNLRPExplainer(BaseExplainer):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, explain_graph=False)

        self.explainer = GNN_LRP(model, explain_graph=False)

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None, **kwargs) -> torch.Tensor:
        if target_class is None:
            target_class = self.get_predicted_label(x, edge_index, edge_weight, node_idx)

        sparsity = kwargs.get('sparsity', 0)
        _, masks, _ = self.explainer(x, edge_index, sparsity=sparsity,
                                     node_idx=node_idx, edge_weight=edge_weight,
                                     given_class=target_class)
        edge_mask = masks[0]
        return self._custom_log_transform(edge_mask)

    @staticmethod
    def _custom_log_transform(x: torch.Tensor) -> torch.Tensor:
        """对 GNN-LRP 的输出进行 log 变换和归一化"""
        log_transformed = torch.empty_like(x)
        finite_mask = torch.isfinite(x)
        positive_mask = (x > 0) & finite_mask
        negative_mask = (x < 0) & finite_mask

        log_transformed[positive_mask] = torch.log(x[positive_mask])
        log_transformed[negative_mask] = -torch.log(-x[negative_mask])
        log_transformed[~finite_mask] = float('-inf')

        valid_vals = log_transformed[finite_mask]
        min_val, max_val = valid_vals.min(), valid_vals.max()

        normalized = torch.zeros_like(x)
        if max_val > min_val:
            normalized[finite_mask] = (log_transformed[finite_mask] - min_val) / (max_val - min_val)
        return normalized


# PGExplainer 解释器
class PGExplainerWrapper(BaseExplainer):
    def __init__(self, model: nn.Module, epochs: int = 100, lr: float = 0.0001, **kwargs):
        super().__init__(model, explain_graph=False)
        self.epochs = epochs
        self.lr = lr

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None, labels=None, **kwargs) -> torch.Tensor:
        from torch_geometric.explain import PGExplainer, Explainer

        explainer = Explainer(
            model=self.model,
            algorithm=PGExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw'
            ),
        )

        # PGExplainer 需要训练
        for epoch in range(self.epochs):
            explainer.algorithm.train(epoch, self.model, x, edge_index,
                                      target=labels, index=node_idx,
                                      edge_weight=edge_weight)

        explanation = explainer(x, edge_index, index=node_idx,
                                edge_weight=edge_weight, target=labels)
        return explanation.edge_mask


# 自定义路径贡献方法解释器
class ConvexExplainer(BaseExplainer):
    """基于路径贡献分解的解释方法"""

    def __init__(self, model: nn.Module, W1: torch.Tensor, W2: torch.Tensor, **kwargs):
        super().__init__(model, explain_graph=False)
        self.W1 = W1
        self.W2 = W2

    def explain(self, x, edge_index, node_idx, edge_weight=None,
                num_classes=None, target_class=None,
                paths=None, target_edgelist=None, sub_adj=None,
                relu_delta=None, relu_start=None, relu_end=None,
                **kwargs) -> torch.Tensor:
        """
        使用路径贡献分解方法生成解释
        需要额外的预计算参数
        """


        # 计算边贡献
        _, _, test_edge_result = test_path_contribution_edge(
            paths, csr_matrix(sub_adj.shape), sub_adj, target_edgelist,
            relu_delta, relu_start, relu_end, x, self.W1, self.W2
        )

        target_edge_result = map_target(test_edge_result, node_idx)

        # 获取模型输出
        with torch.no_grad():
            output = self.model(x, edge_index, edge_index, edge_weight, edge_weight)

        # 计算边的重要性分数
        select_edges_list_value, _ = main_con_edge(
            len(target_edgelist), node_idx, target_edge_result, target_edgelist,
            torch.zeros_like(output[node_idx]).numpy(), output
        )

        return torch.tensor(select_edges_list_value)


# ========== 工厂类 ==========
class ExplainerFactory:
    """解释器工厂类"""

    _explainers = {
        ExplainerMethod.DEEPLIFT: DeepLIFTExplainer,
        ExplainerMethod.FLOWX: FlowXExplainer,
        ExplainerMethod.GNNEXPLAINER: GNNExplainerWrapper,
        ExplainerMethod.GNNLRP: GNNLRPExplainer,
        ExplainerMethod.PGEXPLAINER: PGExplainerWrapper,
        ExplainerMethod.CONVEX: ConvexExplainer,
    }

    @classmethod
    def create(cls, method: str, model: nn.Module, **kwargs) -> BaseExplainer:
        """
        创建解释器实例

        Args:
            method: 方法名称，如 "deeplift", "gnnexplainer" 等
            model: GNN模型
            **kwargs: 解释器特定参数

        Returns:
            解释器实例
        """
        method_enum = ExplainerMethod(method.lower())
        explainer_class = cls._explainers.get(method_enum)

        if explainer_class is None:
            raise ValueError(f"Unknown explainer method: {method}")

        return explainer_class(model, **kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        """返回所有可用的解释方法"""
        return [m.value for m in ExplainerMethod]


# ========== 统一解释管理器 ==========
class UnifiedExplainer:
    """统一的解释器管理器，处理缓存、Ricci曲率等"""

    def __init__(
            self,
            model: nn.Module,
            method: str,
            cache_dir: str = "../result/robustness/edge_masks_normalize",
            dataset_name: str = "cora",
            ricci_lambda: float = 0.0,
            **explainer_kwargs
    ):
        self.model = model
        self.method = method
        self.cache_dir = os.path.join(cache_dir, dataset_name, method)
        self.dataset_name = dataset_name
        self.ricci_lambda = ricci_lambda

        # 创建底层解释器
        self.explainer = ExplainerFactory.create(method, model, **explainer_kwargs)

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

    def explain_node(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            node_idx: int,
            edge_weight: torch.Tensor,
            use_cache: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """
        对单个节点进行解释，支持缓存
        """
        cache_path = os.path.join(self.cache_dir, f"{self.method}_{node_idx}.npy")

        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_path):
            edge_mask_array = np.load(cache_path)
            return torch.tensor(edge_mask_array, dtype=torch.float32)

        # 计算解释
        edge_mask = self.explainer.explain(
            x, edge_index, node_idx, edge_weight, **kwargs
        )

        # 保存到缓存
        if use_cache:
            np.save(cache_path, edge_mask.detach().cpu().numpy())

        return edge_mask

    def explain_with_ricci(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            node_idx: int,
            edge_weight: torch.Tensor,
            ricci_dict: Dict[str, float],
            top_k: int,
            **kwargs
    ) -> Tuple[List, List]:
        """
        结合Ricci曲率的解释，返回基础边和增强边列表
        """
        edge_mask = self.explain_node(x, edge_index, node_idx, edge_weight, **kwargs)

        # 基础边重要性排序
        edge_scores_base = {}
        for i in range(edge_index.size(1)):
            edge_str = f"{edge_index[0, i].item()},{edge_index[1, i].item()}"
            edge_scores_base[edge_str] = edge_mask[i].item()

        # 加入Ricci曲率后的边重要性
        edge_scores_ricci = {}
        for edge_str, score in edge_scores_base.items():
            ricci_score = ricci_dict.get(edge_str, 0.0)
            edge_scores_ricci[edge_str] = score + self.ricci_lambda * ricci_score

        # 选择 top-k 边
        def select_topk(scores_dict: dict, k: int) -> List:
            sorted_edges = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            selected = []
            for edge_str, _ in sorted_edges:
                nodes = [int(n) for n in edge_str.split(',')]
                if nodes not in selected and [nodes[1], nodes[0]] not in selected:
                    selected.append(nodes)
                if len(selected) >= k:
                    break
            return selected

        base_edges = select_topk(edge_scores_base, top_k)
        ricci_edges = select_topk(edge_scores_ricci, top_k)

        return base_edges, ricci_edges