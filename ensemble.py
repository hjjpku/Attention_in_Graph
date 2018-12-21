import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from gcn import *
import gpu
from setproctitle import *
import time
from mlp_dropout import *

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=2,threshold=float('inf'))

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP

from util import cmd_args, load_data,create_process_name
args=cmd_args

pname=create_process_name()
setproctitle(pname+'_ensem')

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
save_path=os.path.join(args.savedir,pname)
log_path=os.path.join(args.logdir,pname+'.txt')
if os.path.exists(save_path):
    os.system('rm -rf '+save_path)
os.makedirs(save_path)
if not args.print:
    f=open(log_path,'a+')
    sys.stderr=f
    sys.stdout=f

gpu.find_idle_gpu(args.gpu)
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.rank_loss=args.rank_loss
        self.model=args.model
        self.eps=args.eps
        if args.pool=='mean':
            self.pool=self.mean_pool
        elif args.pool=='max':
            self.pool=self.max_pool

        if self.model=='gcn':
            self.num_layers=args.gcn_layers
            self.gcns=nn.ModuleList()
            x_size=args.input_dim
            for _ in range(self.num_layers):
                self.gcns.append(GCNBlock(x_size,args.hidden_dim,args.gcn_res,args.gcn_norm,args.dropout,args.relu))
                x_size=args.hidden_dim
            self.mlp=MLPClassifier(args.hidden_dim,args.mlp_hidden,args.num_class,args.mlp_layers,args.dropout)

        elif self.model=='agcn':
            self.margin=args.margin
            self.num_layers=args.num_layers
            assert args.gcn_layers%self.num_layers==0
            args.gcn_layers=args.gcn_layers//self.num_layers
            self.agcns=nn.ModuleList()
            x_size=args.input_dim
            for _ in range(args.num_layers):
                self.agcns.append(AGCNBlock(args,x_size,args.hidden_dim,args.gcn_layers,args.dropout,args.relu))
                x_size=self.agcns[-1].pass_dim
            self.mlps=nn.ModuleList()
            for _ in range(args.num_layers):
                self.mlps.append(MLPClassifier(input_size=args.hidden_dim, hidden_size=args.mlp_hidden, num_class=args.num_class,num_layers=args.mlp_layers,dropout=args.dropout))
        
    def PrepareFeatureLabel(self, batch_graph):
        batch_size = len(batch_graph)
        labels = torch.LongTensor(batch_size)
        max_node_num = 0

        for i in range(batch_size):
            labels[i] = batch_graph[i].label
            max_node_num = max(max_node_num, batch_graph[i].num_nodes)
            #print('tags:',batch_graph[i].node_tags)
        masks = torch.zeros(batch_size, max_node_num)
        adjs =  torch.zeros(batch_size, max_node_num, max_node_num)   

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            batch_node_tag = torch.zeros(batch_size, max_node_num, args.feat_dim)
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            batch_node_feat = torch.zeros(batch_size, max_node_num, args.attr_dim)
        else:
            node_feat_flag = False

        for i in range(batch_size):
            cur_node_num =  batch_graph[i].num_nodes 

            if node_tag_flag == True:
                tmp_tag_idx = torch.LongTensor(batch_graph[i].node_tags).view(-1, 1)
                tmp_node_tag = torch.zeros(cur_node_num, args.feat_dim)
                tmp_node_tag.scatter_(1, tmp_tag_idx, 1)
                batch_node_tag[i, 0:cur_node_num] = tmp_node_tag
            #node attribute feature
            if node_feat_flag == True:
                tmp_node_fea = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

            #adjs
            adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_graph[i].adj

            #masks  
            masks[i,0:cur_node_num] = 1  
            
        #cobime the two kinds of node feature
        if node_feat_flag == True:
            node_feat = batch_node_feat.clone()

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([batch_node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = batch_node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(batch_size,max_node_num,1)  # use all-one vector as node features

        if args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            adjs = adjs.cuda()
            masks = masks.cuda()

        return node_feat, labels, adjs, masks

    def forward(self,batch_graph):
        '''
        node_feat: FloatTensor, [batch,max_node_num,input_dim]
        labels: LongTensor, [batch] 
        adj: FloatTensor, [batch,max_node_num,max_node_num]
        mask: FloatTensor, [batch,max_node_num]
        '''
        node_feat, labels, adj,mask = self.PrepareFeatureLabel(batch_graph)
#        print('node_feat:',node_feat.type(),node_feat.shape,node_feat)
        return self.agcn_forward(node_feat,labels,adj,mask,is_print=False),labels

    def mean_pool(self,x,mask):
        return x.sum(dim=1)/(self.eps+mask.sum(dim=1,keepdim=True))

    @staticmethod
    def max_pool(x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r

    def agcn_forward(self,node_feat,labels,adj,mask,is_print=False):
#        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        X=node_feat
        embeds=[]
        
        for i in range(self.num_layers):
            embed,X,adj,mask,visualize_tool=self.agcns[i](X,adj,mask,is_print=is_print)
            embeds.append(embed)

        return torch.cat(embeds,dim=1)

def loop_dataset(g_list, classifier,ensembler, sample_idxes, epoch,optimizer=None, bsize=50):

    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = range(total_iters)
#    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []
    n_samples = 0
    
    visual_pos=[int(x) for x in args.sample.strip().split(',')]
    
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets

        embeds,labels=classifier(batch_graph)
        logits,_, loss, acc=ensembler(embeds.detach(),labels)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
#        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc,avg_acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().data.numpy()
    
    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))
    
    return avg_loss

def main():

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    classifier = Classifier()
    ensembler = MLPClassifier(args.hidden_dim*3,args.mlp_hidden,args.num_class,num_layers=args.mlp_layers,dropout=args.dropout)
    print(classifier)
    print(ensembler)
    for n,p in classifier.named_parameters():
        print(n,p.type(),p.shape)
    if args.mode == 'gpu':
        classifier = classifier.cuda()
        ensembler = ensembler.cuda()

    classifier.load_state_dict(torch.load(os.path.join(save_path,'best_model.pth')))

    classifier.eval()
    optimizer = optim.Adam(ensembler.parameters(), lr=args.lr)
    
    train_idxes = list(range(len(train_graphs)))
    best_acc=float('-inf')
    best_epoch=0
    for epoch in range(args.epochs):
        start_time=time.time()
        random.shuffle(train_idxes)
        avg_loss = loop_dataset(train_graphs, classifier, ensembler,train_idxes, epoch,optimizer=optimizer,bsize=args.bsize)
        if not args.printAUC:
            avg_loss[3] = 0.0
        print('=====>average training of epoch %d: loss %.5f acc %.5f auc %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))
#        log_value('train acc',avg_loss[1],epoch)

        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))),epoch,bsize=args.test_bsize)
        if best_acc<test_loss[1]:
            best_acc=test_loss[1]
            best_epoch=epoch
            torch.save(ensembler.state_dict(),os.path.join(save_path,'best_ensembler.pth'))

        if not args.printAUC:
            test_loss[3] = 0.0
        print('=====>average test of epoch %d: loss %.5f acc %.5f best acc %.5f(%d) time:%.0fs' % (epoch, test_loss[0], test_loss[1],best_acc,best_epoch,time.time()-start_time))

    if args.printAUC:
        with open('auc_results.txt', 'a+') as f:
            f.write(str(test_loss[-1]) + '\n')
        

if __name__ == '__main__':
    main()

if not args.print:
    f.close()
