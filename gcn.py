import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

import numpy as np

class AGCNBlock(nn.Module):
    def __init__(self,):
        super.__init__(config,input_dim,hidden_dim,output_dim,gcn_layer=2,dropou=0.0)
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.gcns=nn.ModuleList()
        self.gcns.append(GCNBlock(input_dim,hidden_dim,config.gcn_add_self,config.gcn_normalize,dropout))
        for x in self.gcns:
            self.gcns.append(GCNBlock(hidden_dim,hidden_dim,config.gcn_add_self,config.gcn_normalize,dropout))

        self.w_a=nn.Parameter(torch.zeros(1,hidden_dim,1))
        torch.nn.init.normal_(self.w_a)
        
        self.fc=nn.Linear(hidden_dim,output_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias)
        
        self.feat_mode=config.feat_mode
        if config.pool=='mean':
            self.pool=self.mean_pool
        elif config.pool=='max':
            self.pool=self.max_pool

        self.filt_percent=config.percent
        self.eps=config.eps
        

    def forward(self,X,adj,mask):
    '''
    input:
        X:  node input features , [batch,node_num,input_dim],dtype=float
        adj: adj matrix, [batch,node_num,node_num], dtype=float
        mask: mask for nodes, [batch,node_num]
    outputs:
        out:unormalized classification prob, [batch,output_dim]
        H: batch of node hidden features, [batch,node_num,output_dim]
        new_adj: pooled new adj matrix, [batch, k_max, k_max]
        new_mask: [batch, k_max]
    '''
        hidden=X

        for gcn in self.gcns:
            hidden=gcn(hidden,adj)
        
        hidden=mask.unsqueeze(2)*hidden
        out=self.fc(self.pool(hidden,mask))
        
#        att=torch.matmul(hidden,self.w_a).squeeze()
        att=torch.nn.functional.softmax(torch.matmul(hidden,self.w_a).squeeze()+(mask-1)*1e10,dim=1)
        
        if self.feat_mode=='raw':
            Z=X
        elif self.feat_mode=='trans':
            Z=hidden
        elif self.feat_mode=='concat':
            Z=torch.cat([X,hidden],dim=2)
        else:
            raise Exception('unknown output feature mode!')
        Z=att.unsqueeze(2)*Z
        
        k_max=int(math.ceil(self.filt_percent*adj.shape[-1]))
        k_list=torch.ceil(self.filt_percent*mask.sum(dim=1))
        
        _,top_index=torch.topk(att,k_max,dim=1)
        #assign_m: [batch,k_max,node_num]
        new_mask=X.new_zeros((X.shape[0],k_max)
        assign_m=X.new_zeros((X.shape[0],k_max,adj.shape[-1]))
        for i,k in enumerate(k_list):
            for j in range(int(k.item())):
                assign_m[i][j]=adj[i][top_index[i][j]]
                new_mask[i][j]=1.
        assign_m=assign_m/(assign_m.sum(dim=1,keepdim=True)+self.eps)
        
        new_adj=torch.matmul(torch.matmul(assign_m,adj),torch.transpose(assign_m,1,2))
        H=torch.matmul(assign_m,Z)

        return out,H,new_adj,new_mask
    
    @staticmethod
    def mean_pool(x,mask):
        return x.sum(dim=1)/(self.eps+mask.sum(dim=1,keepdim=True))
    
    @staticmethod
    def max_pool(x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r

# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super().__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

