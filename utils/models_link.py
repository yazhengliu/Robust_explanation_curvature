import torch_geometric.nn as gnn
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
from dig.xgraph.models import IdenticalPool
class GCNConvExplainer(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    def forward(self, x: torch.Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> torch.Tensor:
        if self.normalize and edge_weight is None:
            if isinstance(edge_index, torch.Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

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

        elif isinstance(edge_index, torch.Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            if self._explain:
                edge_mask = self.__edge_mask__
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:
                edge_mask = self.layer_edge_mask
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)
            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

class NetLinkEvaluate(torch.nn.Module):
    """用于 DeepLIFT 解释器的模型"""
    def __init__(self, nfeat, nhid):
        super(NetLinkEvaluate, self).__init__()
        self.conv1 = GCNConvExplainer(nfeat, nhid, add_self_loops=False, normalize=False, bias=False)
        self.conv2 = GCNConvExplainer(nhid, nhid, add_self_loops=False, normalize=False, bias=False)
        self.linear = nn.Linear(nhid * 2, 2, bias=False)
        self.readout = IdenticalPool()

    def encode(self, x, edge_index, edge_weight):
        x = self.conv1(x.to(torch.float32), edge_index, edge_weight=edge_weight)
        x = x.relu()
        return self.conv2(x, edge_index, edge_weight=edge_weight)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def forward(self, x, edge_index, edge_weight, pos_edge_index):
        z = self.encode(x, edge_index, edge_weight)
        z = self.readout(z)
        z = self.decode(z, pos_edge_index)
        return z

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)


class NetLinkEvaluatePYG(torch.nn.Module):
    """用于 GNNExplainer/PGExplainer 的模型（使用标准 PyG GCNConv）"""
    def __init__(self, nfeat, nhid):
        super(NetLinkEvaluatePYG, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(nfeat, nhid, add_self_loops=False, normalize=False, bias=False)
        self.conv2 = GCNConv(nhid, nhid, add_self_loops=False, normalize=False, bias=False)
        self.linear = nn.Linear(nhid * 2, 2, bias=False)

    def encode(self, x, edge_index, edge_weight):
        x = self.conv1(x.to(torch.float32), edge_index, edge_weight=edge_weight)
        x = x.relu()
        return self.conv2(x, edge_index, edge_weight=edge_weight)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def forward(self, x, edge_index, edge_weight, pos_edge_index):
        z = self.encode(x, edge_index, edge_weight)
        z = self.decode(z, pos_edge_index)
        return z