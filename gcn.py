import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

import numpy as np

torch.set_printoptions(precision=2,threshold=float('inf'))

class AGCNBlock(nn.Module):
    def __init__(self,config,input_dim,hidden_dim,gcn_layer=2,dropout=0.0,relu=0):
        super(AGCNBlock,self).__init__()
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.model=config.model
        self.gcns=nn.ModuleList()
        self.gcns.append(GCNBlock(input_dim,hidden_dim,config.bn,config.gcn_res,config.gcn_norm,dropout,relu))

        for i in range(gcn_layer-1):
            if i==gcn_layer-2 and (not config.lastrelu):
                self.gcns.append(GCNBlock(hidden_dim,hidden_dim,config.bn,config.gcn_res,config.gcn_norm,dropout,0))
            else:
                self.gcns.append(GCNBlock(hidden_dim,hidden_dim,config.bn,config.gcn_res,config.gcn_norm,dropout,relu))
            
        if self.model=='diffpool':
            self.pool_gcns=nn.ModuleList()
            tmp=input_dim
            self.diffpool_k=config.diffpool_k
            for i in range(config.pool_layers):
                self.pool_gcns.append(GCNBlock(tmp,config.diffpool_k,config.bn,config.gcn_res,config.gcn_norm,dropout,relu))
                tmp=config.diffpool_k

        self.w_a=nn.Parameter(torch.zeros(1,hidden_dim,1))
        self.w_b=nn.Parameter(torch.zeros(1,hidden_dim,1))
        torch.nn.init.normal_(self.w_a)
        torch.nn.init.uniform_(self.w_b,-1,1)

        self.pass_dim=hidden_dim

        if config.pool=='mean':
            self.pool=self.mean_pool
        elif config.pool=='max':
            self.pool=self.max_pool
        elif config.pool=='sum':
            self.pool=self.sum_pool

        self.softmax=config.softmax
        if self.softmax=='gcn':
            self.att_gcn=GCNBlock(2,1,config.gcn_res,config.gcn_norm,dropout,relu)
        self.khop=config.khop
        self.adj_norm=config.adj_norm

        self.filt_percent=config.percent
        self.eps=config.eps

        self.tau_config=config.tau
        if config.tau==-1.:
            self.tau=nn.Parameter(torch.tensor(1),requires_grad=False)
        elif config.tau==-2.:
            self.tau_fc=nn.Linear(hidden_dim,1)
            torch.nn.init.constant_(self.tau_fc.bias,1)
            torch.nn.init.xavier_normal_(self.tau_fc.weight.t())
        else:
            self.tau=nn.Parameter(torch.tensor(config.tau))
        self.lamda1=nn.Parameter(torch.tensor(config.lamda))
        self.lamda2=nn.Parameter(torch.tensor(config.lamda))

        self.att_norm=config.att_norm
        
        self.dnorm=config.dnorm
        self.dnorm_coe=config.dnorm_coe

        self.att_out=config.att_out
        self.single_att=config.single_att


    def forward(self,X,adj,mask,is_print=False):
        '''
    input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
        adj: adj matrix, [batch,node_num,node_num], dtype=float
        mask: mask for nodes, [batch,node_num]
    outputs:
        out:unormalized classification prob, [batch,hidden_dim]
        H: batch of node hidden features, [batch,node_num,pass_dim]
        new_adj: pooled new adj matrix, [batch, k_max, k_max]
        new_mask: [batch, k_max]
        '''
        hidden=X

        if adj.shape[-1]>50:
            is_print=False

        for gcn in self.gcns:
            hidden=gcn(hidden,adj,mask)
        
        hidden=mask.unsqueeze(2)*hidden
        
        if self.model=='unet':
            att=torch.matmul(hidden,self.w_a).squeeze()
            att=att/torch.sqrt((self.w_a.squeeze(2)**2).sum(dim=1,keepdim=True))
        elif self.model=='agcn':
            if self.softmax=='global' or self.softmax=='mix':
                att_a=torch.matmul(hidden,self.w_a).squeeze()+(mask-1)*1e10
                att_a_1=att_a=torch.nn.functional.softmax(att_a,dim=1)
                if self.dnorm:
                    scale=mask.sum(dim=1,keepdim=True)/self.dnorm_coe
                    att_a=scale*att_a
            if self.softmax=='neibor' or self.softmax=='mix':
                att_b=torch.matmul(hidden,self.w_b).squeeze()+(mask-1)*1e10
                att_b_max,_=att_b.max(dim=1,keepdim=True)
                if self.tau_config!=-2:
                    att_b=torch.exp((att_b-att_b_max)*torch.abs(self.tau))
                else:
                    att_b=torch.exp((att_b-att_b_max)*torch.abs(self.tau_fc(self.pool(hidden,mask))))
                denom=att_b.unsqueeze(2)
                for _ in range(self.khop):
                    denom=torch.matmul(adj,denom)
                denom=denom.squeeze()+self.eps
                att_b=att_b/denom
                if self.dnorm:
                    diag_scale=mask/(torch.diagonal(adj,0,1,2)+self.eps)/self.dnorm_coe
                    att_b=att_b*diag_scale
                
            elif self.softmax=='hardnei':
                denom=adj
                for _ in range(self.khop-1):
                    denom=torch.matmul(adj,denom)
                denom=(denom>0).type_as(att_b)
                denom=torch.matmul(denom,att_b.unsqueeze(2)).squeeze()+self.eps
                att_b=att_b/denom

            if self.softmax=='global':
                att=att_a
            elif self.softmax=='neibor' or self.softmax=='hardnei':
                att=att_b
            elif self.softmax=='mix':
                att=att_a*torch.abs(self.lamda1)+att_b*torch.abs(self.lamda2)
            elif self.softmax=='gcn':
                att=torch.stack([att_a,att_b],dim=2)
                if self.att_norm:
                    att=torch.nn.functional.normalize(att,dim=1)
                att=self.att_gcn(att,adj)
                att=torch.nn.functional.softmax(att.squeeze(2),dim=1)
                
        Z=hidden

        if self.model=='unet':
            Z=torch.tanh(att.unsqueeze(2))*Z
        elif self.model=='agcn':
            Z=att.unsqueeze(2)*Z
        

        k_max=int(math.ceil(self.filt_percent*adj.shape[-1]))
        if self.model=='diffpool':
            k_max=min(k_max,self.diffpool_k)

        k_list=[int(math.ceil(self.filt_percent*x)) for x in mask.sum(dim=1).tolist()]

        if self.model!='diffpool':
            _,top_index=torch.topk(att,k_max,dim=1)

        new_mask=X.new_zeros(X.shape[0],k_max)

        visualize_tools=None 

        if self.model=='unet':
            for i,k in enumerate(k_list):
                for j in range(int(k),k_max):
                    top_index[i][j]=adj.shape[-1]-1
                    new_mask[i][j]=-1.
            new_mask=new_mask+1
            top_index,_=torch.sort(top_index,dim=1)
            assign_m=X.new_zeros(X.shape[0],k_max,adj.shape[-1])
            for i,x in enumerate(top_index):
                assign_m[i]=torch.index_select(adj[i],0,x)
            new_adj=X.new_zeros(X.shape[0],k_max,k_max)
            H=Z.new_zeros(Z.shape[0],k_max,Z.shape[-1])
            for i,x in enumerate(top_index):
                new_adj[i]=torch.index_select(assign_m[i],1,x)
                H[i]=torch.index_select(Z[i],0,x)

        elif self.model=='agcn':
            assign_m=X.new_zeros(X.shape[0],k_max,adj.shape[-1])
            for i,k in enumerate(k_list):
                for j in range(int(k)):
                    assign_m[i][j]=adj[i][top_index[i][j]]
                    new_mask[i][j]=1.
            assign_m=assign_m/(assign_m.sum(dim=1,keepdim=True)+self.eps)

            H=torch.matmul(assign_m,Z)
            
            new_adj=torch.matmul(torch.matmul(assign_m,adj),torch.transpose(assign_m,1,2))

        elif self.model=='diffpool':
            hidden1=X
            for gcn in self.pool_gcns:
                hidden1=gcn(hidden1,adj,mask)
            assign_m=X.new_ones(X.shape[0],X.shape[1],k_max)*(-100000000.)
            for i,x in enumerate(hidden1):
                k=min(k_list[i],k_max)
                assign_m[i,:,0:k]=hidden1[i,:,0:k]
                for j in range(int(k)):
                    new_mask[i][j]=1.

            assign_m=torch.nn.functional.softmax(assign_m,dim=2)*mask.unsqueeze(2)
            assign_m_t=torch.transpose(assign_m,1,2)
            new_adj=torch.matmul(torch.matmul(assign_m_t,adj),assign_m)
            H=torch.matmul(assign_m_t,Z)
            
        if self.att_out:
            if self.single_att:
                out=self.pool(hidden,mask)
            else:
                out=self.pool(att_a_1.unsqueeze(2)*hidden,mask)
        else:
            out=self.pool(hidden,mask)
            
        if self.adj_norm=='tanh' or self.adj_norm=='mix':
            new_adj=torch.tanh(new_adj)
        elif self.adj_norm=='diag' or self.adj_norm=='mix':
            diag_elem=torch.pow(new_adj.sum(dim=2)+self.eps,-0.5)
            diag=new_adj.new_zeros(new_adj.shape)
            for i,x in enumerate(diag_elem):
                diag[i]=torch.diagflat(x)
            new_adj=torch.matmul(torch.matmul(diag,new_adj),diag)


        visualize_tools=[]
        if (not self.training) and is_print:

            print('**********************************')
            print('node_feat:',X.type(),X.shape)
            print(X)

            if self.model!='diffpool':
                print('**********************************')
                print('att:',att.type(),att.shape)
                print(att)
                visualize_tools.append(att[0])

                print('**********************************')
                print('top_index:',top_index.type(),top_index.shape)
                print(top_index)
                visualize_tools.append(top_index[0])


            print('**********************************')
            print('adj:',adj.type(),adj.shape)
            print(adj)

            print('**********************************')
            print('assign_m:',assign_m.type(),assign_m.shape)
            print(assign_m)

            print('**********************************')
            print('new_adj:',new_adj.type(),new_adj.shape)
            print(new_adj)
            visualize_tools.append(new_adj[0])

            print('**********************************')
            print('new_mask:',new_mask.type(),new_mask.shape)
            print(new_mask)
            visualize_tools.append(new_mask.sum())
            
        return out,H,new_adj,new_mask,visualize_tools
    
    def mean_pool(self,x,mask):
        return x.sum(dim=1)/(self.eps+mask.sum(dim=1,keepdim=True))
    
    def sum_pool(self,x,mask):
        return x.sum(dim=1)

    @staticmethod
    def max_pool(x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r
# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0,add_self=0, normalize_embedding=0,
            dropout=0.0,relu=0, bias=True):
        super(GCNBlock,self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu=relu
        self.bn=bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = masked_batchnorm(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, mask):
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
        y = mask.unsqueeze(2)*y
        if self.bn:
            y=self.bn_layer(y,mask)
        if self.relu:
            y=torch.nn.functional.relu(y)
        return y

class masked_batchnorm(nn.Module):
    def __init__(self,feat_dim,epsilon=1e-10):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(feat_dim))
        self.beta=nn.Parameter(torch.zeros(feat_dim))
        self.eps=epsilon

    def forward(self,x,mask):
        '''
        x: node feat, [batch,node_num,feat_dim]
        mask: [batch,node_num]
        '''
        mask1 = mask.unsqueeze(2)
        mask_sum = mask.sum()
        mean = x.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        temp = (x - mean)**2
        temp = temp*mask1
        var = temp.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        rstd = torch.rsqrt(var+self.eps)
        x=(x-mean)*rstd
        return ((x*self.alpha) + self.beta)*mask1
