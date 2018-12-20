from __future__ import print_function
import sys
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
cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-gpu', default='',type=str, help='gpu number')
cmd_opt.add_argument('-name', default='train', help='')
cmd_opt.add_argument('-print', type=int, default=0, help='')
cmd_opt.add_argument('-logdir', default='log', help='')
cmd_opt.add_argument('-savedir', default='save', help='')
cmd_opt.add_argument('-save', default=1, help='whether to save running metadata')
cmd_opt.add_argument('-save_freq', default=10, help='to save running metadata')
cmd_opt.add_argument('-sample', default='0,1,2', type=str,help='sample test minibatch for visulization')
cmd_opt.add_argument('-data', default='NCI1', help='data folder name')
cmd_opt.add_argument('-bsize', type=int, default=20, help='minibatch size')
cmd_opt.add_argument('-test_bsize', type=int, default=1, help='test minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-epochs', type=int, default=500, help='number of epochs')
cmd_opt.add_argument('-lr', type=float, default=1e-3, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=float, default=0., help='')
cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')


#classifier options:
cls_opt=cmd_opt.add_argument_group('classifier options')
cls_opt.add_argument('-model', type=str, default='agcn', help='model choice:gcn/agcn')
cls_opt.add_argument('-hidden_dim', type=int, default=64, help='hidden size k')
cls_opt.add_argument('-num_class', type=int, default=1000, help='classification number')
cls_opt.add_argument('-num_layers', type=int, default=3, help='layer number of agcn block')
cls_opt.add_argument('-mlp_hidden', type=int, default=100, help='hidden size of mlp layers')
cls_opt.add_argument('-mlp_layers', type=int, default=2, help='layer numer of mlp layers')
cls_opt.add_argument('-eps', type=float, default=1e-20, help='')

gcn_opt=cmd_opt.add_argument_group('gcn options')
gcn_opt.add_argument('-gcn_res', type=int, default=0, help='whether to use residual structure in gcn layers')
gcn_opt.add_argument('-gcn_norm', type=int, default=0, help='whether to normalize gcn layers')
gcn_opt.add_argument('-relu', type=int, default=1, help='whether to use relu')
gcn_opt.add_argument('-gcn_layers', type=int, default=6, help='layer number in each agcn block')

gcn_opt.add_argument('-att_norm', type=int, default=0, help='layer number in each agcn block')

#agcn options:
agcn_opt=cmd_opt.add_argument_group('agcn options')
agcn_opt.add_argument('-feat_mode', type=str,default='trans', help='whether to normalize gcn layers: a)raw:output raw feature b)trans:output gcn feature c)concat:output concatenation of raw and gcn feature ')
agcn_opt.add_argument('-pool', type=str,default='mean',help='agcn pool method: mean/max')
agcn_opt.add_argument('-softmax', type=str,default='global',help='agcn pool method: global/neighbor')
agcn_opt.add_argument('-khop', type=int,default=1,help='agcn pool method: global/neighbor')
agcn_opt.add_argument('-adj_norm', type=str,default='none',help='agcn pool method: global/neighbor')
agcn_opt.add_argument('-rank_loss', type=int,default=0,help='agcn pool method: global/neighbor')
agcn_opt.add_argument('-margin', type=float, default=0.05, help='margin value in rank loss')
agcn_opt.add_argument('-percent', type=float,default=0.5,help='agcn node keep percent(=k/node_num)')
agcn_opt.add_argument('-tau', type=float,default=1.,help='agcn node keep percent(=k/node_num)')
agcn_opt.add_argument('-lamda', type=float,default=1.,help='agcn node keep percent(=k/node_num)')


cmd_args = cmd_opt.parse_args()

#cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
#if len(cmd_args.latent_dim) == 1:
#    cmd_args.latent_dim = cmd_args.latent_dim[0]

class Hjj_Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features  
        '''
        super().__init__()
        self.num_nodes = len(node_tags)
        self.node_tags = self.__rerange_tags(node_tags, list(g.nodes())) # rerangenodes index
        self.label = label
        self.g=g
        self.node_features = self.__rerange_fea(node_features, list(g.nodes()))  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree()).values()) # type(g.degree()) is dict 
        self.adj = self.__preprocess_adj(nx.adjacency_matrix(g)) # torch.FloatTensor
        
    def __rerange_fea(self, node_features, node_list):
        if node_features == None or node_features == []:
            return node_features
        else:
            new_node_features = []
            for i in range(node_features.shape[0]):
                new_node_features.append(node_features[node_list[i]])

            new_node_features = np.vstack(new_node_features)
            return new_node_features
   
    def __rerange_tags(self, node_tags, node_list):
        new_node_tags = []
        if node_tags != []:
            for i in range(len(node_tags)):
                new_node_tags.append(node_tags[node_list[i]])

        return new_node_tags


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



def create_process_name():
    argvs=sys.argv[1:]
    tmp=[]
    has_model=0
    has_data=0
    for x in argvs:
        if ('-model=' in x) or ('-data=' in x):
            continue
        n,v=x.strip('-').split('=')
        tmp.append(n+'^'+v)
    name='_'.join(tmp)
    name='_'.join([cmd_args.model,name,cmd_args.data])
    return name
