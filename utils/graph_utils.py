from torch_geometric.utils import degree
import torch
import torch_geometric.transforms as T
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
import copy
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
    """初始化没有节点特征的图"""
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


def rumor_construct_adj_matrix(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    return adj

def rumor_construct_adj_matrix_v2(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(adj)
    return adj


def matrixtodict(nonzero):
    """将邻接矩阵转换为字典形式"""
    a = []
    graph = dict()
    for i in range(0, len(nonzero[1])):
        if i != len(nonzero[1]) - 1:
            if nonzero[0][i] == nonzero[0][i + 1]:
                a.append(nonzero[1][i])
            if nonzero[0][i] != nonzero[0][i + 1]:
                a.append(nonzero[1][i])
                graph[nonzero[0][i]] = a
                a = []
        if i == len(nonzero[1]) - 1:
            a.append(nonzero[1][i])
        graph[nonzero[0][len(nonzero[1]) - 1]] = a
    return graph


def clear(edges):
    """去除重复边"""
    edge_clear = []
    for idx, edge in enumerate(edges):
        if [edge[0], edge[1]] not in edge_clear and [edge[1], edge[0]] not in edge_clear:
            edge_clear.append([edge[0], edge[1]])
    return edge_clear


def dfs2(start, index, graph, length, path=[], paths=[]):
    """找到起点长度定值的全部路径"""
    path.append(index)
    if len(path) == length:
        paths.append(path.copy())
        path.pop()
    else:
        for item in graph[index]:
            dfs2(start, item, graph, length, path, paths)
        path.pop()
    return paths


def reverse_paths(pathlist):
    """反转路径列表"""
    result = []
    for path in pathlist:
        path.reverse()
        result.append(path)
    return result


def softmax(x):
    """计算 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def KL_divergence(P, Q, eps=1e-5):
    """计算 KL 散度"""
    P = np.abs(P - eps) if 0 in P else P
    Q = np.abs(Q - eps) if 0 in Q else Q
    return sum(P * np.log(P / Q))


def edge_percentage(x):
    """计算边掩码的集中度"""
    if x.isnan().all():
        return 1.0
    x = x[~x.isnan()]
    values, counts = x.unique(return_counts=True)
    max_count = counts.max()
    percentage = max_count.item() / x.numel()
    return percentage


def custom_log_transform(x):
    """对边掩码进行 log 变换和归一化"""
    log_transformed = torch.empty_like(x)
    finite_mask = torch.isfinite(x)
    positive_mask = (x > 0) & finite_mask
    negative_mask = (x < 0) & finite_mask

    log_transformed[positive_mask] = torch.log(x[positive_mask])
    log_transformed[negative_mask] = -torch.log(-x[negative_mask])
    log_transformed[~finite_mask] = float('-inf')

    valid_vals = log_transformed[finite_mask]
    min_val = valid_vals.min()
    max_val = valid_vals.max()

    normalized = torch.zeros_like(x)
    if max_val - min_val > 0:
        normalized[finite_mask] = (log_transformed[finite_mask] - min_val) / (max_val - min_val)

    return normalized


def normalize_ricci(curv):
    """归一化 Ricci 曲率到 [0, 1]"""
    return (curv + 1) / 2


def from_edges_to_evaulate(select_edges_list, edges_weight_old, edges_old, edges_old_dict, adj_old, adj_new):
    """将选择的边转换为评估用的边索引和权重"""
    evaluate_edge_weight = copy.deepcopy(edges_weight_old.tolist()) if hasattr(edges_weight_old, 'tolist') else list(edges_weight_old)
    evaluate_edge_index = copy.deepcopy(edges_old)

    for edge in select_edges_list:
        if edge[0] != edge[1]:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]
                value2 = edges_old_dict[str(edge[1]) + ',' + str(edge[0])]
                evaluate_edge_weight[value2] = adj_new[edge[1], edge[0]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])
                evaluate_edge_index[0].append(edge[1])
                evaluate_edge_index[1].append(edge[0])
                evaluate_edge_weight.append(adj_new[edge[1], edge[0]])
        else:
            if adj_old[edge[0], edge[1]] != 0:
                value1 = edges_old_dict[str(edge[0]) + ',' + str(edge[1])]
                evaluate_edge_weight[value1] = adj_new[edge[0], edge[1]]
            else:
                evaluate_edge_index[0].append(edge[0])
                evaluate_edge_index[1].append(edge[1])
                evaluate_edge_weight.append(adj_new[edge[0], edge[1]])

    return evaluate_edge_index, evaluate_edge_weight

def test_path_contribution_edge(paths,adj_start,adj_end,addedgelist,relu_delta,relu_start,relu_end,x_tensor,W1,W2):
    XW1=torch.mm(x_tensor,W1)
    path_result_dict=dict()
    node_result_dict=dict()
    edge_result_dict_zong=dict()
    for edge in addedgelist:
        edge_key=str(edge[0])+','+str(edge[1])
        edge_result_dict_zong[edge_key]=np.zeros((adj_end.shape[0],W2.shape[1]))


    for path in paths:
        if ([path[0],path[1]] in addedgelist or  [path[1],path[0]] in addedgelist) and ([path[2],path[1]] in addedgelist or  [path[1],path[2]] in addedgelist):
            # print(adj_end[path[1],path[2]])


            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                f1=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)
                f3=f1+f2

                f4=f1
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

            else:
                # print('torch.mul',torch.mul(relu_delta[path[1]],XW1[path[0]]))
                # weight=(adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]])
                # print(weight.shape)
                # print(W2.shape)
                # print('f1',torch.mm(torch.unsqueeze(weight,0),W2))
                f1=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*torch.mul(relu_delta[path[1]],XW1[path[0]]),0),W2)
                f2=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0)
                    , W2)
                f3=f1+f2

                f4=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)
                f5=(adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_start[path[1],path[0]])*relu_start[path[1]]*XW1[path[0]],0),W2)

                # f1 = adj_start[path[1]][path[2]] * np.dot(
                #     (adj_end[path[1]][path[0]] - adj_start[path[1]][path[0]]) * relu_delta[path[1]] * XW1[path[0]], W2)
                # f2 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_end[path[1]][path[0]]) * relu_end[path[1]] * XW1[path[0]], W2)
                # f3 = f1 + f2
                #
                # f4 = f1
                # f5 = (adj_end[path[1]][path[2]] - adj_start[path[1]][path[2]]) * np.dot(
                #     (adj_start[path[1]][path[0]]) * relu_start[path[1]] * XW1[path[0]], W2)

            # print('f1',f1)
            # print('f2', f2)
            # print('f3',f3)
            f3=torch.squeeze(f3,0)
            f4 = torch.squeeze(f4, 0)
            f5 = torch.squeeze(f5, 0)
            f3=f3.detach().numpy()
            f4=f4.detach().numpy()
            f5=f5.detach().numpy()

            contribution_edge_1 = 0.5 * (f3 - f5 + f4)
            contribution_edge_2 = 0.5 * (f3 - f4 + f5)

            # print('contribution_edge_1',contribution_edge_1)
            # print('contribution_edge_1.shape', contribution_edge_1.shape)

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = f3
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = f3
            else:
                node_result_dict[path[2]] += f3


            if [path[0],path[1]] in addedgelist:
                edge_1=str(path[0])+','+str(path[1])
            else:
                edge_1 = str(path[1]) + ',' + str(path[0])

            if edge_1 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_1][path[2]]+=contribution_edge_1
            else:
                edge_result_dict_zong[edge_1][path[2]]= contribution_edge_1

            if [path[2], path[1]] in addedgelist:
                edge_2 = str(path[2]) + ',' + str(path[1])
            else:
                edge_2 = str(path[1]) + ',' + str(path[2])

            if edge_2 in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_2][path[2]] += contribution_edge_2
            else:
                edge_result_dict_zong[edge_2][path[2]] = contribution_edge_2



        else:
            for i in range(len(path) - 1):
                # edge_key = str(path[i]) + ',' + str(path[i + 1]) +','+ str(i)
                if [path[i], path[i + 1]] in addedgelist:
                    edge_key = str(path[i]) + ',' + str(path[i + 1])
                elif [path[i + 1], path[i]] in addedgelist:
                    edge_key = str(path[i + 1]) + ',' + str(path[i])

            if adj_end[path[1],path[2]]>=adj_start[path[1],path[2]]:
                contribution=adj_start[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) *torch.mm(torch.unsqueeze((adj_end[path[1],path[0]])*relu_end[path[1]]*XW1[path[0]],0),W2)

            else:
                contribution=adj_end[path[1],path[2]]*torch.mm(torch.unsqueeze((adj_end[path[1],path[0]]-adj_start[path[1],path[0]])*relu_delta[path[1]]*XW1[path[0]],0),W2)+ \
                             (adj_end[path[1],path[2]] - adj_start[path[1],path[2]]) * torch.mm(
                    torch.unsqueeze((adj_start[path[1],path[0]]) * relu_start[path[1]] * XW1[path[0]],0), W2)






            # if edge_key in edge_result_dict_zong.keys():
            #     edge_result_dict_zong[edge_key][path[2]]+=contribution
            # else:
            #     edge_result_dict_zong[edge_key][path[2]]= contribution
            contribution=torch.squeeze(contribution,0)
            contribution=contribution.detach().numpy()

            if edge_key in edge_result_dict_zong.keys():
                edge_result_dict_zong[edge_key][path[2]]+=contribution
            else:
                edge_result_dict_zong[edge_key][path[2]]= contribution

            p_key = str(path[0]) + ',' + str(path[1]) + ',' + str(path[2])
            path_result_dict[p_key] = contribution
            if path[2] not in node_result_dict.keys():
                node_result_dict[path[2]] = contribution
            else:
                node_result_dict[path[2]] += contribution

    return path_result_dict,node_result_dict,edge_result_dict_zong

def main_con_edge(select_number_path,edge_result_dict, edgelist,gcn_old_tensor,output_new): #convex

    edge_selected= cvx.Variable(len(edgelist),integer=False)

    tmp_logits = copy.deepcopy(gcn_old_tensor)

    # print('edge_result_dict',edge_result_dict)
    # print('edgelist',edgelist)

    for i in range(len(edgelist)):
        add_matrix = np.array(
            edge_result_dict[str(edgelist[i][0]) + ',' + str(edgelist[i][1])])
        tmp_logits = tmp_logits + edge_selected[i] * add_matrix

    # print(old_tensor.shape)
    # print('tmp_logits ',tmp_logits)

    tmp_logits = cvx.sum(tmp_logits, axis=0)/gcn_old_tensor.shape[0]

    new_prob=softmax(output_new)
    d=0
    for i in range(0,2):
        d=d+tmp_logits[i]*new_prob[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(tmp_logits))
    constraints = [sum(edge_selected)== select_number_path]

    for i in range(0,len(edgelist)):
        constraints.append(0 <= edge_selected[i])
        constraints.append(edge_selected[i] <= 1)
    # print('constraints',constraints)
    prob = cvx.Problem(objective, constraints)
    solver_options = {'max_iters': 1000, 'eps': 1e-4}
    #
    prob.solve(solver='MOSEK',warm_start=True,mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  500.0,
                                    }) #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.value)
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    edge_res = []
    # group1_res = m.getAttr(group1_selected)
    # print('group0_res =', group0_res)

    for i in range(len(edgelist)):
        edge_res.append(
            edge_selected[i].value)

    #print('edge_res', edge_res)

    # result0 = [i for i, x in enumerate(edge_res) if abs(x - 1) < 1e-4]
    # print('result0',result0)

    sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

    # select_edges_list = []
    # for i in range(select_number_path):
    #     select_edges_list.append([edgelist[sorted_id[i]][0], edgelist[sorted_id[i]][1]])

        # print('edge contribution',edge_res[sorted_id[i]])

    # print('select_edges_list', select_edges_list)

    return edge_res,sorted_id

def normalize_ricci(kappa):
    return (kappa + 2) / 3

def smooth(arr, eps=1e-5):
    if 0 in arr:
        return abs(arr - eps)
    else:
        return arr

def KL_divergence(P, Q):
    # Input P and Q would be vector (like messages or priors)
    # Will calculate the KL-divergence D-KL(P || Q) = sum~i ( P(i) * log(Q(i)/P(i)) )
    # Refer to Wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    P = smooth(P)
    Q = smooth(Q)
    return sum(P * np.log(P / Q))