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
from gcn1 import *
#from tensorboard_logger import configure,log_value
import gpu
from setproctitle import *
import time

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=2,threshold=float('inf'))

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP

from util import cmd_args, load_data,create_process_name
args=cmd_args
if args.init_from!='':
    tmp=args.init_from
    state_dict=torch.load(args.init_from)
    args=state_dict['args']
    args.init_from=tmp

pname=create_process_name()
setproctitle(pname)

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

train_graphs, test_graphs = load_data()
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
                self.gcns.append(GCNBlock(x_size,args.hidden_dim,args.bn,args.gcn_res,args.gcn_norm,args.dropout,args.relu))
                x_size=args.hidden_dim
            self.mlp=MLPClassifier(args.hidden_dim,args.mlp_hidden,args.num_class,args.mlp_layers,args.dropout)

        else:
            self.margin=args.margin
            self.agcn_res=args.agcn_res
            self.single_loss=args.single_loss
            self.num_layers=args.num_layers
            assert args.gcn_layers%self.num_layers==0
            args.gcn_layers=args.gcn_layers//self.num_layers


            self.agcns=nn.ModuleList()
            x_size=args.input_dim

            for _ in range(args.num_layers):
                self.agcns.append(AGCNBlock(args,x_size,args.hidden_dim,args.gcn_layers,args.dropout,args.relu))
                x_size=self.agcns[-1].pass_dim
                if args.model=='diffpool':
                    args.diffpool_k=int(math.ceil(args.diffpool_k*args.percent))
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

    def forward(self,batch_graph,is_print=False):
        '''
        node_feat: FloatTensor, [batch,max_node_num,input_dim]
        labels: LongTensor, [batch] 
        adj: FloatTensor, [batch,max_node_num,max_node_num]
        mask: FloatTensor, [batch,max_node_num]
        '''
        node_feat, labels, adj,mask = self.PrepareFeatureLabel(batch_graph)
#        print('node_feat:',node_feat.type(),node_feat.shape,node_feat)
#        print('labels:',labels.type(),labels.shape,labels)
#        print('adj:',labels.type(),adj.shape,adj)
#        print('mask:',labels.type(),mask.shape,mask)
        if self.model=='gcn':
            return self.gcn_forward(node_feat,labels,adj,mask)
        else:
            return self.agcn_forward(node_feat,labels,adj,mask,is_print=is_print)

    def mean_pool(self,x,mask):
        return x.sum(dim=1)/(self.eps+mask.sum(dim=1,keepdim=True))

    @staticmethod
    def max_pool(x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r

    def gcn_forward(self,node_feat,labels,adj,mask):
        X=node_feat
        for i in range(self.num_layers):
            X=self.gcns[i](X,adj,mask)
        embed=self.pool(X,mask)
        logits,_,loss,acc=self.mlp(embed,labels)
        return logits,loss,acc,acc,None
        
    def agcn_forward(self,node_feat,labels,adj,mask,is_print=False):
#        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        
        cls_loss=node_feat.new_zeros(self.num_layers)
        rank_loss=node_feat.new_zeros(self.num_layers-1)
        X=node_feat
        p_t=[]
        pred_logits=0
        visualize_tools=[]
        embeds=0

        for i in range(self.num_layers):
            embed,X,adj,mask,visualize_tool=self.agcns[i](X,adj,mask,is_print=is_print)
            embeds=embeds+embed

            visualize_tools.append(visualize_tool)

            if not self.agcn_res:
                logits,softmax_logits,loss,acc=self.mlps[i](embed,labels)
            else:
                logits,softmax_logits,loss,acc=self.mlps[i](embeds,labels)

            '''
            cls_loss=loss
            '''
            pred_logits=pred_logits+softmax_logits
            cls_loss[i]=loss

            if self.rank_loss:
                p_mask=softmax_logits.new_zeros(softmax_logits.shape,dtype=torch.uint8)
                for j,cls in enumerate(labels):
                    p_mask[j][cls]=1
                p_t.append(torch.masked_select(softmax_logits,p_mask))
                if i>0:
                    tmp=p_t[i-1]-p_t[i]+self.margin
                    rank_loss[i-1]=torch.max(tmp,torch.zeros_like(tmp)).mean()

        pred=pred_logits.data.max(1)[1]
        avg_acc = pred.eq(labels.data.view_as(pred)).cpu().sum().item() / float(labels.size()[0])

        if is_print:
            if self.training:
                print('training sample loss')
            else:
                print('test sample loss')
            print('cls_loss:',cls_loss)
            print('rank_loss:',rank_loss)
        
        if self.single_loss:
            cls_loss=cls_loss[-1]
            avg_acc=acc
        if self.rank_loss:
            loss=cls_loss.mean()+rank_loss.mean()
        else:
            loss=cls_loss.mean()
        return logits,loss,acc,avg_acc,visualize_tools

def loop_dataset(g_list, classifier, sample_idxes, epoch,optimizer=None, bsize=50):

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

        if (not classifier.training) and (pos in visual_pos) and args.model!='gcn':
            print('=======================test minibatch:',pos,'==================================')

        logits, loss, acc,avg_acc,visualize_tools = classifier(batch_graph,is_print=(pos in visual_pos))
        all_scores.append(logits[:, 1].detach())  # for binary classification

        if epoch%args.save_freq==0 and (not classifier.training) and args.save and (pos in visual_pos) and args.model!='gcn':
            visualize_tools=list(zip(*visualize_tools))
            visualize_tools=[[x.detach().cpu().numpy() for x in y] for y in visualize_tools]
            np.save(os.path.join(save_path,'sample%03d_epoch%03d.npy'%(pos,epoch)),[batch_graph[0].g,batch_graph[0].node_tags]+visualize_tools)
            

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),args.max_grad_norm)
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
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
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

    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))


    classifier = Classifier()
    print(classifier)
    for n,p in classifier.named_parameters():
        print(n,p.type(),p.shape)
    if args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    
    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    best_acc=float('-inf')
    best_avg_acc=float('-inf')
    best_overall_acc=float('-inf')
    start_epoch=0
    best_epoch=0
    best_avg_epoch=0

    if args.init_from!='':
        classifier.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        start_epoch=state_dict['epoch']
        best_overall_acc=state_dict['best_overall_acc']

        dummy_idxes=list(range(len(train_graphs)))
        for _ in range(start_epoch):
            random.shuffle(dummy_idxes)
            
        
    p=0
    for epoch in range(start_epoch+1,args.epochs):
        if args.decay==1 and epoch==args.epochs-args.patient:
            for pg in optimizer.param_groups:
                tmp=pg['lr']=pg['lr']*0.1
            print('===>>lr decay to %f'%tmp)
        if args.decay==2 and p>=args.patient:
            for pg in optimizer.param_groups:
                tmp=pg['lr']=pg['lr']*0.1
            print('===>>lr decay to %f'%tmp)
            p=0

        start_time=time.time()
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, epoch,optimizer=optimizer,bsize=args.bsize)
        if not args.printAUC:
            avg_loss[3] = 0.0
        print('=====>average training of epoch %d: loss %.5f acc %.5f avg_acc %.5f auc %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2],avg_loss[3]))
#        log_value('train acc',avg_loss[1],epoch)

        classifier.eval()

        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))),epoch,bsize=args.test_bsize)
        if best_acc<test_loss[1]:
            best_acc=test_loss[1]
            best_epoch=epoch
        if best_avg_acc<test_loss[2]:
            best_avg_acc=test_loss[2]
            best_avg_epoch=epoch
            p=0
        else:
            p+=1
        if max(best_acc,best_avg_acc)>best_overall_acc:
            best_overall_acc=max(best_acc,best_avg_acc)
            torch.save({'model_state_dict':classifier.state_dict(),
                'optim_state_dict':optimizer.state_dict(),
                'args':args,
                'epoch':epoch,
                'best_overall_acc':best_overall_acc},
                os.path.join(save_path,'best_model.pth'))

        if not args.printAUC:
            test_loss[3] = 0.0
        print('=====>average test of epoch %d: loss %.5f acc %.5f avg_acc %.5f best acc %.5f(%d) %.5f(%d) time:%.0fs' % (epoch, test_loss[0], test_loss[1],test_loss[2], best_acc,best_epoch,best_avg_acc,best_avg_epoch,time.time()-start_time))
        if args.model=='agcn' and args.tau>0:
            for k in range(classifier.num_layers):
                print('layer%d: tau=%.5f, lamda1=%.5f lamda2=%.5f'%(k,classifier.agcns[k].tau.item(),classifier.agcns[k].lamda1.item(),classifier.agcns[k].lamda2.item()))


    if args.printAUC:
        with open('auc_results.txt', 'a+') as f:
            f.write(str(test_loss[-1]) + '\n')

    with open(os.path.join(args.logdir,'acc_results.txt'), 'a+') as f:
        f.writelines(pname+': '+'%.4f(%d) %.4f(%d)'%(best_acc,best_epoch,best_avg_acc,best_avg_epoch))
        

if __name__ == '__main__':
    main()

if not args.print:
    f.close()
