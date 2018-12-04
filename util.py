from __future__ import print_function
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import pdb
import argparse
import scipy.sparse as sp

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-attr_dim', type=int, default=0, help='dimension of continues node attribute (node feature)')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=float, default=0.5, help='')
cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')


#classifier options:
cls_opt=cmd_opt.add_argument_group('classifier options')
cls_opt.add_argument('-model', type=str, default='agcn', help='model choice:gcn/agcn')
cls_opt.add_argument('-input_dim', type=int, default=1, help='input dimension of node features')
cls_opt.add_argument('-hidden_dim', type=int, default=64, help='hidden size k')
cls_opt.add_argument('-num_class', type=int, default=1000, help='classification number')
cls_opt.add_argument('-num_layers', type=int, default=3, help='layer number of agcn block')
cls_opt.add_argument('-mlp_hidden', type=int, default=100, help='hidden size of mlp layers')
cls_opt.add_argument('-mlp_layers', type=int, default=2, help='layer numer of mlp layers')
cls_opt.add_argument('-margin', type=int, default=0.05, help='margin value in rank loss')
cls_opt.add_argument('-eps', type=int, default=1e-10, help='')

#gcn options:
gcn_opt=cmd_opt.add_argument_group('gcn options')
gcn_opt.add_argument('-gcn_add_self', type=int, default=1, help='whether to use residual structure in gcn layers')
gcn_opt.add_argument('-gcn_norm', type=int, default=1, help='whether to normalize gcn layers')
gcn_opt.add_argument('-gcn_layers', type=int, default=2, help='layer number in each agcn block')

#agcn options:
agcn_opt=cmd_opt.add_argument_group('agcn options')
agcn_opt.add_argument('-feat_mode', type=str,default='trans', help='whether to normalize gcn layers: a)raw:output raw feature b)trans:output gcn feature c)concat:output concatenation of raw and gcn feature ')
agcn_opt.add_argument('-pool', type=str,default='max',help='agcn pool method: mean/max')
agcn_opt.add_argument('-percent', type=float,default=0.3,help='agcn node keep percent(=k/node_num)')


cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

class Hjj_Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features  
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree()).values()) # type(g.degree()) is dict 
        self.adj = self.__preprocess_adj(nx.adjacency_matrix(g)) # torch.FloatTensor

    def __sparse_to_tensor(self, adj):
        '''
            adj: sparse matrix in COOrdinate format
        '''
        assert sp.isspmatrix_coo(adj), 'not coo format sparse matrix'
            #adj = adj.tocoo()

        values = adj.data
        indices = np.vstack((adj.row, adj.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    

    def __normalize_adj(self, sp_adj):
        adj = sp.coo_matrix(sp_adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


    def __preprocess_adj(self, sp_adj):
        '''
            sp_adj: sparse matrix in Compressed Sparse Row format
        '''
        adj_normalized = self.__normalize_adj(sp_adj + sp.eye(sp_adj.shape[0]))
        return self.__sparse_to_tensor(adj_normalized)

def load_data():

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip()) # number of graph
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] #node number & graph label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            #assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(Hjj_Graph(g, l, node_tags, node_features))
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict) # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1] # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0
    cmd_args.input_dim = cmd_args.feat_dim + cmd_args.attr_dim


    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    else:
        return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number :]



