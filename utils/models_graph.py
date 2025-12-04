import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size
import torch
import torch.nn as nn
import torch.nn.functional as F
from dig.xgraph.models import GlobalMeanPool
from torch_sparse import SparseTensor

class GCNConv(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        edge_weight.requires_grad_(True)
        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out += self.bias

        self.edge_weight = edge_weight
        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        if (isinstance(edge_index, SparseTensor) and self.fuse and not self._explain):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)
            msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            if self._explain:
                edge_mask = self.__edge_mask__
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)
            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


class GCN_Graph(torch.nn.Module):
    """图级别 GCN 模型，带 readout"""

    def __init__(self, nfeat, hidden_channels, nclass):
        super(GCN_Graph, self).__init__()
        self.conv1 = GCNConv(nfeat, hidden_channels, add_self_loops=False, normalize=False, bias=False)
        self.conv2 = GCNConv(hidden_channels, nclass, add_self_loops=False, normalize=False, bias=False)
        self.readout = GlobalMeanPool()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.readout(x, batch)
        return x

    def pre_forward(self, x, edge_index, edge_weight):
        """返回 readout 前的节点嵌入"""
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def back(self, x, edge_index, edge_weight):
        """返回中间层激活值"""
        x_0 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index, edge_weight=edge_weight)
        return x_0, x_1

    def verify_layeredge(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2):
        """用于验证的分层前向传播"""
        x = F.relu(self.conv1(x, edge_index1, edge_weight=edge_weight1))
        x = self.conv2(x, edge_index2, edge_weight=edge_weight2)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.readout(x, batch)
        return x


from torch_geometric.nn import GCNConv as GCNConvPyG
from torch_geometric.nn import global_mean_pool

class GCN_Graph_pyg(torch.nn.Module):
    """图级别 GCN 模型，使用标准 PyG GCNConv（用于 GNNExplainer, PGExplainer）"""

    def __init__(self, nfeat, hidden_channels, nclass):
        super(GCN_Graph_pyg, self).__init__()
        self.conv1 = GCNConvPyG(nfeat, hidden_channels, add_self_loops=False, normalize=False, bias=False)
        self.conv2 = GCNConvPyG(hidden_channels, nclass, add_self_loops=False, normalize=False, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return x

    def pre_forward(self, x, edge_index, edge_weight=None):
        """返回 readout 前的节点嵌入"""
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def back(self, x, edge_index, edge_weight=None):
        """返回中间层激活值"""
        x_0 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index, edge_weight=edge_weight)
        return x_0, x_1

    def verify_layeredge(self, x, edge_index1, edge_index2, edge_weight1, edge_weight2):
        """用于验证的分层前向传播"""
        x = F.relu(self.conv1(x, edge_index1, edge_weight=edge_weight1))
        x = self.conv2(x, edge_index2, edge_weight=edge_weight2)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        return x