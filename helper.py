from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


def orthogonal(shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

def orthogonal_initializer(shape, scale=1.0, dtype=torch.FloatTensor):
        return torch.Tensor(orthogonal(shape) * scale).type(dtype)
        

class layernorm(nn.Module):
    def __init__(self,num_units,base,epsilon=1e-3):
        super(layernorm,self).__init__()
        self.alpha=nn.Parameter(torch.ones(base*num_units))
        self.beta=nn.Parameter(torch.zeros(base*num_units))
        self.base=base
        self.num_units=num_units
        self.epsilon=epsilon

    def forward(self,h):
        h_reshape=h.view([-1,self.base,self.num_units])
        mean = h_reshape.mean(dim = 2,keepdim=True)
        temp = (h_reshape - mean)**2
        var = temp.mean(dim = 2,keepdim=True)
        rstd = torch.rsqrt(var+self.epsilon)
        h_reshape=(h_reshape-mean)*rstd
        h = h_reshape.view([-1, self.base * self.num_units])
        return (h*self.alpha) + self.beta

#class zoneout(nn.Module):
#    def __init__(self,h_keep,c_keep):
#        super().__init__()
#        self.c_dropout=nn.Dropout(p=1-c_keep)
#        self.h_dropout=nn.Dropout(p=1-h_keep)
#        self.h_keep=h_keep
#        self.c_keep=c_keep
#    
#    def forward(self,new_h, new_c, h, c):
#        mask_c = torch.ones_like(c)
#        mask_h = torch.ones_like(h)
#
#        mask_c = self.c_dropout(mask_c)
#        mask_h = self.h_dropout(mask_h)
#
#        h = new_h * mask_h + (-mask_h + 1.) * h
#        c = new_c * mask_c + (-mask_c + 1.) * c
#
#        return h, c
#
#class zoneout1(nn.Module):
#    def __init__(self,c_keep):
#        super().__init__()
#        self.c_dropout=nn.Dropout(p=1-c_keep)
#        self.c_keep=c_keep
#    
#    def forward(self, new_c,  c):
#        mask_c = torch.ones_like(c)
#
#        mask_c = self.c_dropout(mask_c)
#
#        c = new_c * mask_c + (-mask_c + 1.) * c
#
#        return  c
##
class zoneout(nn.Module):
    def __init__(self,h_keep,c_keep):
        super().__init__()
        self.c_dropout=nn.Dropout(p=1-c_keep)
        self.h_dropout=nn.Dropout(p=1-h_keep)
        self.h_keep=h_keep
        self.c_keep=c_keep
    
    def forward(self,new_h, new_c, h, c):
        mask_c = torch.ones_like(c)
        mask_h = torch.ones_like(h)

        mask_c = self.c_dropout(mask_c)
        mask_h = self.h_dropout(mask_h)

        mask_c *= self.c_keep
        mask_h *= self.h_keep

        h = new_h * mask_h + (-mask_h + 1.) * h
        c = new_c * mask_c + (-mask_c + 1.) * c

        return h, c

class zoneout1(nn.Module):
    def __init__(self,c_keep):
        super().__init__()
        self.c_dropout=nn.Dropout(p=1-c_keep)
        self.c_keep=c_keep
    
    def forward(self, new_c,  c):
        mask_c = torch.ones_like(c)

        mask_c = self.c_dropout(mask_c)

        mask_c *= self.c_keep

        c = new_c * mask_c + (-mask_c + 1.) * c

        return  c
