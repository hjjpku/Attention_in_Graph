import torch
import util
import visualize

file_name = 'save/agcn_softmax^global_adj_norm^none_name^normal_eps^1e-10_NCI1/sample000_epoch040.vis'
a=torch.load(file_name)
print(a[3])
visualize(a[0],a[1],a[2],a[3],a[4])


