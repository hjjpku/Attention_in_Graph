from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)

        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers=2,dropout=0.):
        super(MLPClassifier, self).__init__()

        self.num_layers=num_layers
        if self.num_layers==2:
            self.h1_weights = nn.Linear(input_size, hidden_size)
            self.h2_weights = nn.Linear(hidden_size, num_class)
            torch.nn.init.xavier_normal(self.h1_weights.weight)
            torch.nn.init.constant(self.h1_weights.bias,0)
            torch.nn.init.xavier_normal(self.h2_weights.weight)
            torch.nn.init.constant(self.h2_weights.bias,0)
        elif self.num_layers==1:
            self.h1_weights = nn.Linear(input_size,num_class) 
            torch.nn.init.xavier_normal(self.h1_weights.weight)
            torch.nn.init.constant(self.h1_weights.bias,0)
        self.dropout=dropout
        if self.dropout>0.001:
            self.dropout_layer=nn.Dropout(p=dropout)


    def forward(self, x, y = None):
        if self.num_layers==2:
            h1 = self.h1_weights(x)
            h1 = F.relu(h1)
            if self.dropout>0.001:
                h1=self.dropout_layer(h1)

            logits = self.h2_weights(h1)
        elif self.num_layers==1:
            logits=self.h1_weights(x)
        
        softmax_logits = F.softmax(logits,dim=1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum() / float(y.size()[0])
            #acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits,softmax_logits, loss, acc
        else:
            return logits
