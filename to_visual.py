import torch
import util
from visualize import visualize

file_name = 'save/agcn_softmax^global_adj_norm^none_COLLAB/sample000_epoch000.vis'
a=torch.load(file_name)
visualize(*a)


