from torch_geometric.data import Data,InMemoryDataset, DataLoader
import os.path as osp
import os
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import time,datetime
import math
import torch
import argparse
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch import Tensor
import cvxpy as cvx
import copy
class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data = link_read_data(self.root, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

def split_edge(start,end,flag,clear_time,num_nodes):
    edge_index = [[], []]
    max_month=0
    max_week=0
    print('start',start)
    print('end',end)
    if flag == 'year':
        for key, value in clear_time.items():
            if value[0] >= start and value[0] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    if flag == 'month':
        for key, value in clear_time.items():
            # print('key',key,'value',value)
            max_month=max(max_month,value[1])
            if value[1] >= start and value[1] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])

    if flag=='week':
        for key, value in clear_time.items():
            max_month = max(max_month, value[1])
            max_week = max(max_week, value[2])
            if value[2] >= start and value[2] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    # print('max_month',max_month)
    # print('max_week', max_week)

    # for i in range(num_nodes):
    #     edge_index[0].append(i)
    #     edge_index[1].append(i)
    return edge_index

def clear_time(time_dict):
    edge_time = dict()
    max_month=0

    for key, value in time_dict.items():
        month = (value.year - 2010) * 12 + value.month
        max_month=max(month,max_month)
        week = (value.year - 2010) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    print('max_month',max_month)
    return clear_time

def clear_time_UCI(time_dict):
    edge_time = dict()
    max_week = 0

    for key, value in time_dict.items():
        # print('value.year',value.year)
        month = (value.year - 2004) * 12 + value.month
        week = (value.year - 2004) * 52 + value.isocalendar()[1]
        max_week = max(week, max_week)
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    # print('max_week',max_week)
    return clear_time

def link_read_data(folder: str, prefix):
    # path=os.path.join(folder, f"{prefix}.npz")
    # data_csv = 'bitcoinotc.csv'
    path = os.path.join(folder, f"{prefix}.csv")
    # print(path)
    edges_index,X ,mapping,time_dict= link_load_data(path)

    # x = torch.from_numpy(features).float()
    # y = torch.from_numpy(labels)
    # print('y',y)
    features=torch.DoubleTensor(X)

    edge_index = torch.LongTensor(edges_index)
    print('ed',edge_index)
    data = Data(x=features,  edge_index=edge_index,node_map=mapping,time_dict=time_dict)
    # node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    # node_mask= torch.zeros(adj.shape[0], dtype=torch.bool)
    # train_mask=node_mask.clone()
    # val_mask = node_mask.clone()
    # test_mask = node_mask.clone()
    # print('node_mask',node_mask)
    # for i in range(0,adj.shape[0]):
    #     if i in idx_train:
    #         # print('train')
    #         train_mask[i]=True
    #     if i in idx_test:
    #         # print('test')
    #         test_mask[i]=True
    #     if i in idx_val:
    #         # print('val')
    #         val_mask[i]=True
    #
    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask
    return data

def link_load_data(path):
    # data_dir = 'data'
    # data_csv = 'bitcoinotc.csv'
    # filename = os.path.join(data_dir, data_csv)
    df = pd.read_csv(path)
    # print(df)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)

    mapping = {}
    count = 0
    for node in list(G.nodes):
        mapping[node] = count
        count = count + 1
    G = nx.relabel_nodes(G, mapping)

    rating = nx.get_edge_attributes(G, 'RATING')
    # print('rating',rating)
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)
    # print(A)
    # if (6002,6000) in G.edges():
    #     print('yes')
    # else:
    #     print('false')

    # print(len(G.edges()))
    # print(len(edges_index[0]))

    feat_dict = {}
    feature_length = 8
    for node in list(G.nodes):
        out_edges_list = G.out_edges(node)
        # print('out_edges_list',out_edges_list)

        if len(out_edges_list) == 0:
            features = np.ones(feature_length, dtype=float) / 1000
            feat_dict[node] = {'feat': features}
        else:
            features = np.zeros(feature_length, dtype=float)
            w_pos = 0
            w_neg = 0
            for (_, target) in out_edges_list:
                w = G.get_edge_data(node, target)['RATING']
                if w >= 0:
                    w_pos = w_pos + w
                else:
                    w_neg = w_neg - w

            abstotal = (w_pos + w_neg)
            average = (w_pos - w_neg) / len(out_edges_list) / max_rating

            features[0] = w_pos / max_rating / len(out_edges_list)  # average positive vote
            features[1] = w_neg / max_rating / len(out_edges_list)  # average negative vote
            features[2] = w_pos / abstotal
            features[3] = average
            features[4] = features[0] * G.in_degree(node) / dmax_in
            features[5] = features[1] * G.in_degree(node) / dmax_in
            features[6] = features[0] * G.out_degree(node) / dmax_out
            features[7] = features[1] * G.out_degree(node) / dmax_out

            features = features / 1.01 + 0.001

            feat_dict[node] = {'feat': features}
    nx.set_node_attributes(G, feat_dict)
    G = G.to_undirected()
    # print(G.edges())
    A = nx.adjacency_matrix(G).todense()
    X = np.asarray([G.nodes[node]['feat'] for node in list(G.nodes)])
    edges_index = [[], []]
    for edge in G.edges():
        edges_index[0].append(edge[0])
        edges_index[1].append(edge[1])
        edges_index[1].append(edge[0])
        edges_index[0].append(edge[1])
    # for i in range(max(edges_index[1])+1):
    #     edges_index[0].append(i)
    #     edges_index[1].append(i)

    time_dict = dict()
    df = df.values
    for i in range(0, df.shape[0]):
        edge_0 = df[i][0]
        edge_1 = df[i][1]
        t1 = datetime.datetime.utcfromtimestamp(df[i][3])
        time_dict[(mapping[edge_0], mapping[edge_1])] = t1

    return edges_index,X,mapping,time_dict #边 节点特征
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
    # print('adj',adj)
    # adj=normalize(adj)
    return adj

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
    # print('adj',adj)
    adj=normalize(adj)
    return adj

class gen_link_data():
    def __init__(self, dataset,data_path,start1,end1,flag,layernumbers):
        self.dataset=dataset
        self.data_path=data_path
        self.start1 = start1
        self.end1 = end1
        self.flag = flag
        self.layernumbers=layernumbers


    def load_data(self):
        dataset = SynGraphDataset(self.data_path,self.dataset)
        modelname = self.dataset
        data = dataset[0]
        time_dict = data.time_dict
        if self.dataset=='UCI':
            clear_time_dict = clear_time_UCI(time_dict)
        else:
            clear_time_dict = clear_time(time_dict)
        # print('clear_time_dict',clear_time_dict)
        # print('data.num_nodes',data.num_nodes)


        edge_time_result = dict()
        for key,value in clear_time_dict.items():
            if value[1] not in edge_time_result.keys():
                edge_time_result[value[1]]=1
            else:
                edge_time_result[value[1]] =edge_time_result[value[1]]+1
        # print('edge_time_result',edge_time_result)
        #
        # print('sort',sorted(edge_time_result.items(), key=lambda x: x[1],reverse=True))

        edge_index_old = split_edge(self.start1, self.end1, self.flag, clear_time_dict,data.num_nodes)









        edge_index_old=torch.tensor(edge_index_old)


        return dataset, edge_index_old,clear_time_dict

    def gen_new_edge(self,target_edge,evaulate_model,edges_all,time_dict,features,normalize_flag):
        goal_1 = target_edge[0]
        goal_2 = target_edge[1]

        pos_edge_index = [[goal_1], [goal_2]]
        evaulate_model.eval()

        subset_1_all, edge_index_1_all, _, _ = k_hop_subgraph(
            goal_1, self.layernumbers, edges_all, relabel_nodes=False,
            num_nodes=None)
        # print(subset, edge_index)
        # print(subset_1,edge_index_1)

        subset_2_all, edge_index_2_all, _, _ = k_hop_subgraph(
            goal_2, self.layernumbers, edges_all, relabel_nodes=False,
            num_nodes=None)



        edges_all_dict = dict()

        edge_index_all= [[], []]
        count = 0
        for i in range(len(edge_index_1_all[0])):
            key = str(edge_index_1_all[0][i].item()) + ',' + str(edge_index_1_all[1][i].item())
            edges_all_dict[key] = count
            edge_index_all[0].append(edge_index_1_all[0][i].item())
            edge_index_all[1].append(edge_index_1_all[1][i].item())
            count += 1
        # print('edges_all_dict',edges_all_dict)

        for i in range(len(edge_index_2_all[0])):
            key = str(edge_index_2_all[0][i].item()) + ',' + str(edge_index_2_all[1][i].item())
            if key not in edges_all_dict.keys():
                edges_all_dict[key] = count
                edge_index_all[0].append(edge_index_2_all[0][i].item())
                edge_index_all[1].append(edge_index_2_all[1][i].item())





        all_node_list = list(set(subset_2_all.tolist()).union(set(subset_1_all.tolist())))

        #print('time_dict',time_dict)

        edge_time_result = dict()
        for i in range(len(edge_index_all[0])):
            node1 = edge_index_all[0][i]
            node2 = edge_index_all[1][i]

            if (node1,node2) not in edge_time_result.keys() and (node2,node1) not in edge_time_result.keys():
                edge_time_result[(node1,node2)]=time_dict[(node1,node2)][2]

        # print('edge_time_result',edge_time_result)

        sort_edge_time_result = sorted(edge_time_result.items(), key=lambda x: x[1])

        sliding_T = math.floor(len(sort_edge_time_result) / 10)

        edge_index_old = split(0, sliding_T*3, sort_edge_time_result,
                               all_node_list)

        # edge_index_old = split(0, len(sort_edge_time_result), sort_edge_time_result,
        #                                               all_node_list)





        submapping, reverse_mapping, map_edge_index_old= subadj_map_v2(
            all_node_list, edge_index_old)



        #print('all_node_list', len(all_node_list))









        # for i in range(len(all_node_list)):
        #     key = str(i) + ',' + str(i)
        #     if key not in map_edge_old_dict.keys():
        #         map_edge_index_old[0].append(i)
        #         map_edge_index_old[1].append(i)
        #         map_edge_old_dict[key] = len(map_edge_old_dict)
        #         # print('',key,map_edge_old_dict[key])
        #         map_edge_weight_old.append(adj_old[reverse_mapping[i], reverse_mapping[i]])
        #
        #     if key not in map_edge_new_dict.keys():
        #         map_edge_index_new[0].append(i)
        #         map_edge_index_new[1].append(i)
        #         map_edge_new_dict[key] = len(map_edge_new_dict)
        #         map_edge_weight_new.append(adj_new[reverse_mapping[i], reverse_mapping[i]])

        # print('map_edge_old_dict',map_edge_old_dict)

        if normalize_flag:
            sub_old = rumor_construct_adj_matrix(map_edge_index_old, len(submapping))
            adj_old_nonzero = sub_old.nonzero()
        else:
            sub_old = rumor_construct_adj_matrix_v2(map_edge_index_old, len(submapping))
            adj_old_nonzero = sub_old.nonzero()




        map_edge_old_dict = dict()
        map_edge_weight_old=[]
        for i in range(len(map_edge_index_old[0])):
            map_edge_old_dict[str(map_edge_index_old[0][i]) + ',' + str(map_edge_index_old[1][i])] = i
            map_edge_weight_old.append(sub_old[map_edge_index_old[0][i], map_edge_index_old[1][i]])



        graph_old = matrixtodict(adj_old_nonzero)


        # print('graph_old',graph_old)



        map_edge_old_dict_reverse = dict()
        for key, value in map_edge_old_dict.items():
            map_edge_old_dict_reverse[value] = key



        sub_features = subfeaturs(features, reverse_mapping)
        sub_features = torch.tensor(sub_features)
        sub_features = sub_features.to(torch.float32)


        map_edge_index_old = torch.tensor(map_edge_index_old)
        map_edge_weight_old = torch.tensor(map_edge_weight_old)
        map_edge_weight_old = map_edge_weight_old.to(torch.float32)

        # print('map_edge_weight_old', map_edge_weight_old)
        # print('sub_features', sub_features)


        return sub_features, sub_old,  map_edge_index_old,  graph_old, \
            map_edge_weight_old, map_edge_old_dict,  submapping,map_edge_old_dict_reverse






    def gen_model(self,data):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=50,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--mlp_hidden', type=int, default=4,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--data.x_size', type=float, default=16,
                            help='Dropout rate (1 - keep probability).')

        args = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model= Net_link(data.num_features,args.hidden).to(device)
        model.eval()
        data_prefix = self.data_path+'/'
        model.load_state_dict(torch.load(data_prefix+ 'GCN_model_without_self'+self.dataset+'.pth'))

        W1 = model.state_dict()['conv1.lin.weight'].t()
        W2 = model.state_dict()['conv2.lin.weight'].t()

        W3=model.state_dict()['linear.weight'].t()

        # print(model.state_dict().keys())
        return model,W1,W2,W3

    def gen_evaulate_model(self,model,data):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=50,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--mlp_hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--data.x_size', type=float, default=16,
                            help='Dropout rate (1 - keep probability).')

        args = parser.parse_args()

        model_gnn = Net_link_evaulate(nfeat=data.num_features,
                             nhid=args.hidden,
                             )
        model_gnn.eval()

        model_dict = model_gnn.state_dict()
        print('model parameters ',model_dict.keys())
        model_dict['conv1.lin.weight'] = model.state_dict()['conv1.lin.weight']
        model_dict['conv2.lin.weight'] = model.state_dict()['conv2.lin.weight']
        model_dict['linear.weight']=model.state_dict()['linear.weight']
        model_gnn.load_state_dict(model_dict)
        return model_gnn




    def gen_parameters(self,model, features,edges_old_tensor,edges_new_tensor,edgeweight1,edgeweight2):
        model.eval()
        nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(features, edges_old_tensor, edges_old_tensor,edgeweight1, edgeweight1)
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_new_tensor, edges_new_tensor,
                                                                     edgeweight2, edgeweight2)

        # print('nonlinear_start_layer1',nonlinear_start_layer1.shape)
        # print('nonlinear_end_layer1', nonlinear_end_layer1.shape)
        #
        # print('nonlinear_start_layer1', nonlinear_relu_start_layer1.shape)
        # print('nonlinear_end_layer1',  nonlinear_relu_end_layer1.shape)

        relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                                 (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                         nonlinear_end_layer1 - nonlinear_start_layer1),
                                 torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
        relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                               torch.zeros_like(nonlinear_end_layer1))
        relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                                 torch.zeros_like(nonlinear_start_layer1))
        return relu_delta, relu_end, relu_start

    def gen_parameters_v2(self,model,features,edges_new_tensor,edgeweight2):
        model.eval()

        # nonlinear_start_layer1, nonlinear_relu_start_layer1 = model.back(features, edges_old_tensor, edges_old_tensor,
        #                                                                  edgeweight1, edgeweight1)
        nonlinear_end_layer1, nonlinear_relu_end_layer1 = model.back(features, edges_new_tensor,edges_new_tensor,
                                                                     edgeweight2,edgeweight2)
        nonlinear_start_layer1=torch.zeros_like(nonlinear_end_layer1)
        nonlinear_relu_start_layer1=torch.zeros_like(nonlinear_relu_end_layer1)

        # print('nonlinear_start_layer1',nonlinear_start_layer1.shape)
        # print('nonlinear_end_layer1', nonlinear_end_layer1.shape)
        #
        # print('nonlinear_start_layer1', nonlinear_relu_start_layer1.shape)
        # print('nonlinear_end_layer1',  nonlinear_relu_end_layer1.shape)

        relu_delta = torch.where((nonlinear_end_layer1 - nonlinear_start_layer1) != 0,
                                 (nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1) / (
                                             nonlinear_end_layer1 - nonlinear_start_layer1),
                                 torch.zeros_like((nonlinear_relu_end_layer1 - nonlinear_relu_start_layer1)))
        relu_end = torch.where((nonlinear_end_layer1) != 0, nonlinear_relu_end_layer1 / nonlinear_end_layer1,
                               torch.zeros_like(nonlinear_end_layer1))
        relu_start = torch.where((nonlinear_start_layer1) != 0, nonlinear_relu_start_layer1 / nonlinear_start_layer1,
                                 torch.zeros_like(nonlinear_start_layer1))

        return relu_delta,relu_end,relu_start


class Net_link(torch.nn.Module):
    def __init__(self,nfeat,nhid):
        super(Net_link, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(nhid, nhid,add_self_loops=False,normalize=False,bias=False)
        self.linear = nn.Linear(nhid * 2, 2,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2, args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)

    def encode(self, x,edge_index1,edge_index2,edge_weight1,edge_weight2):
        # print(type(data.x))
        #
        # print(data.x.type())

        x = self.conv1(x.to(torch.float32), edge_index1,edge_weight=edge_weight1)
        x = x.relu()
        return self.conv2(x, edge_index2,edge_weight=edge_weight2)



    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h=self.linear(h)
        # h = self.MLP1(h)
        # h = h.relu()
        # h = self.MLP2(h)

        # h=h.sum(dim=-1)
        # print('h', h.shape)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # print('logits.shape',logits.shape)
        return h

    def forward(self,x, edge_index1,edge_index2,edge_weight1,edge_weight2,pos_edge_index):
        z = self.encode(x, edge_index1,edge_index2,edge_weight1,edge_weight2)
        z=self.decode(z,pos_edge_index)
        return z

    def back_MLP(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)


class Net_link_evaulate(torch.nn.Module):
    def __init__(self,nfeat,nhid):
        super(Net_link_evaulate, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid,add_self_loops=False,normalize=False,bias=False)
        self.conv2 = GCNConv(nhid, nhid,add_self_loops=False,normalize=False,bias=False)
        self.linear = nn.Linear(nhid * 2, 2,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2, args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)

    def encode(self, x,edge_index,edge_weight):
        # print(type(data.x))
        #
        # print(data.x.type())

        x = self.conv1(x.to(torch.float32), edge_index,edge_weight=edge_weight)
        x = x.relu()
        return self.conv2(x, edge_index,edge_weight=edge_weight)



    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h=self.linear(h)
        # h = self.MLP1(h)
        # h = h.relu()
        # h = self.MLP2(h)

        # h=h.sum(dim=-1)
        # print('h', h.shape)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # print('logits.shape',logits.shape)
        return h

    def forward(self,x, edge_index,edge_weight,pos_edge_index):
        z = self.encode(x, edge_index,edge_weight)
        z=self.decode(z,pos_edge_index)
        return z

    def back_MLP(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h

    def back(self, x, edge_index_1, edge_index_2, edgeweight1, edgeweight2):
        x_0 = self.conv1(x, edge_index_1, edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        return (x_0, x_1)



def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)
    # print('node_idx',node_idx)
    #print('node_mask',node_mask)
    # print('edge_mask', edge_mask)
    # print(len(row))
    inv = None

    subsets = [node_idx]

    for _ in range(num_hops):
        # print(num_hops)
        node_mask.fill_(False)
        # print(node_mask)
        node_mask[subsets[-1]] = True
        # print(row)
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # print(torch.index_select(node_mask, 0, row, out=edge_mask))
        subsets.append(col[edge_mask])
    # print(subsets)
    # print('lensubsets',len(subsets[3]))
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    # print('inv',inv)
    # print('subset',subset)
    inv = inv[:node_idx.numel()]
    # print(subsets)
    # print('inv', inv)
    # print(node_idx.numel())

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    # print(edge_mask)
    # print(len(edge_mask))

    edge_index = edge_index[:, edge_mask]
    # print(len(edge_index[0]))
    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def split(start,end,edge_result,all_nodes):
    edges_index = [[], []]
    for i in range(start,end):
        # print(edge_result[i][0])
        # print(edge_result[i][1])

        node1=int(edge_result[i][0][0])
        node2 = int(edge_result[i][0][1])

        edges_index[0].append(node1)
        edges_index[1].append(node2)

        edges_index[1].append(node1)
        edges_index[0].append(node2)


    for i in all_nodes:
        edges_index[0].append(i)
        edges_index[1].append(i)
    return edges_index

def subadj_map_v2(union, edge_index_old):
    mapping = dict()
    for it, neighbor in enumerate(union):
        mapping[neighbor] = it

    # print('mapping',mapping)

    flipped_mapping = {v: k for k, v in mapping.items()}

    map_edge_index_old=[[],[]]


    for i in range(len(edge_index_old[0])):
        map_edge_index_old[0].append(mapping[edge_index_old[0][i]])
        map_edge_index_old[1].append(mapping[edge_index_old[1][i]])




    return mapping,flipped_mapping,map_edge_index_old

def matrixtodict(nonzero): # 将邻接矩阵变为字典形式存储
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

def subfeaturs(features,mapping):
    subarray = np.array(np.zeros((len(mapping), features.shape[1])))
    for i in range(len(mapping)):
        subarray[i]=features[mapping[i]]
    return subarray

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def dfs2(start,index,graph,length,path=[],paths=[]):#找到起点长度定值的全部路径
    path.append(index)
    # print('index',index)
    # if length==0:
    #     return paths
    if len(path)==length:
        paths.append(path.copy())
        path.pop()
    else:
        for item in graph[index]:
            # if item not in path:
                dfs2(start,item,graph,length,path,paths)
        path.pop()

    return paths

def reverse_paths(pathlist):
    result=[]
    for path in pathlist:
        path.reverse()
        result.append(path)
    return result

def clear(edges):
    edge_clear=[]
    for idx,edge in enumerate(edges):
        # if idx%1000==0:
        #     # print('idx',idx)
        if [edge[0],edge[1]] not in edge_clear and [edge[1],edge[0]] not in edge_clear:
            edge_clear.append([edge[0],edge[1]])
    return edge_clear

def edge_percentage(x, tol=0.001):
    if x.isnan().all():
        return 1.0
    x = x[~x.isnan()]
    if x.numel() == 0:
        return 1.0

    x_sorted, _ = x.sort()
    groups = [x_sorted[0].item()]
    counts = [1]

    for val in x_sorted[1:]:
        if abs(val.item() - groups[-1]) < tol:
            counts[-1] += 1
        else:
            groups.append(val.item())
            counts.append(1)

    max_count = max(counts)
    percentage = max_count / x.numel()
    return percentage

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

def map_target(result_dict,target_node):
    final_dict=dict()
    for key,value in result_dict.items():
        final_dict[key]=value[target_node]
    return final_dict

def mlp_contribution(result_dict,W):
    for key,value in result_dict.items():
        result_dict[key]=value.dot(W)
    return result_dict

def main_con_edge(select_number_path,edge_result_dict, edgelist,old_tensor,new_tensor): #convex

    edge_selected= cvx.Variable(len(edgelist),boolean=False)

    tmp_logits = copy.deepcopy(old_tensor)

    # print('edge_result_dict',edge_result_dict)
    # print('edgelist',edgelist)

    for i in range(len(edgelist)):
        add_matrix = np.array(
            edge_result_dict[str(edgelist[i][0]) + ',' + str(edgelist[i][1])])
        tmp_logits = tmp_logits + edge_selected[i] * add_matrix

    # print(old_tensor.shape)
    # print('tmp_logits ',tmp_logits)

    new_prob=softmax(new_tensor)
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

    # for i in range(0,len(edgelist)):
    #     constraints.append(0 <= edge_selected[i])
    #     constraints.append(edge_selected[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    solver_options = {'max_iters': 1000, 'eps': 1e-4}
    #
    prob.solve(solver='MOSEK', warm_start=True, mosek_params={'MSK_DPAR_OPTIMIZER_MAX_TIME': 500.0,
                                                              })  # solver=
    # print('x.value',x.value)  # A numpy ndarray.**
    edge_res = []
    # group1_res = m.getAttr(group1_selected)
    # print('group0_res =', group0_res)

    for i in range(len(edgelist)):
        edge_res.append(
            edge_selected[i].value)

    #print('edge_res', edge_res)

    # print('edge_res', edge_res)

    # result0 = [i for i, x in enumerate(edge_res) if abs(x - 1) < 1e-4]
    # print('result0',result0)

    # sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)
    #
    # select_edges_list = []
    # for i in range(select_number_path):
    #     select_edges_list.append([edgelist[sorted_id[i]][0], edgelist[sorted_id[i]][1]])

    sorted_id = sorted(range(len(edge_res)), key=lambda k: edge_res[k], reverse=True)

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