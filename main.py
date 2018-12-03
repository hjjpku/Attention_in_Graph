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

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP

from util import cmd_args, load_data
args=cmd_args

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.margin=args.margin
        self.num_layers=args.num_layers
        self.agcns=nn.ModuleList()
        self.agcns.append(AGCNBlock(args,args.input_dim,args.hidden_size,args.gcn_layers,args.dropout))
        for _ in range(args.num_layers-1):
            pass_dim=self.agcns[-1].pass_dim
            self.agcns.append(AGCNBlock(args,pass_dim,args.hidden_dim,args.gcn_layers,args.dropout))
        self.mlps=nn.ModuleList()
        for _ in range(args.num_layers):
            self.mlps.append(MLPClassifier(input_size=args.hidden_dim, hidden_size=args.mlp_hidden, num_class=args.num_class,num_layers=args.mlp_layers,dropout=args.dropout))
        
    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
#        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        '''
        node_feat: FloatTensor, [batch,node_num,input_dim]
        labels: LongTensor, [batch] 
        adj: FloatTensor, [batch,node_num,node_num]
        mask: FloatTensor, [batch,node_num]
        '''
        node_feat, labels, adj,mask = self.PrepareFeatureLabel(batch_graph)
        
        cls_loss=0
        rank_loss=0
        X=node_feat
        p_t=[]
        for i in range(self.num_layers):
            embed,X,adj,mask=self.agcns[i](X,adj,mask)
            logits,softmax_logits,loss,acc=self.mlps[i](embed,labels)
            cls_loss=cls_loss+loss
            #?
            p_mask=torch.ByteTensor(1).new_zeros(softmax_logits.shape)
            for i,cls in enumerate(labels):
                p_mask[i][cls]=1
            p_t.append(torch.masked_select(softmax_logits,p_mask))
            if i>0:
                tmp=p_t[i]-p_t[i-1]+self.margin
                rank_loss=rank_loss+torch.max(tmp,torch.zeros_like(tmp)).mean()

        return logits,cls_loss+rank_loss,acc

    def output_features(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return embed, labels
        

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().data.numpy()
    print(type(all_scores))
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))
    
    return avg_loss


if __name__ == '__main__':
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        args.sortpooling_k = num_nodes_list[int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1]
        args.sortpooling_k = max(10, args.sortpooling_k)
        print('k used in SortPooling is: ' + str(args.sortpooling_k))

    classifier = Classifier()
    if args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

    with open('acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if args.printAUC:
        with open('auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
