import os
import numpy as np
import random


def find_idle_gpu(id=''):
    if id=='': 
        tmp_file='tmp_'+str(random.randint(0,10000))
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >'+tmp_file)
        memory_gpu=[int(x.split()[2]) for x in open(tmp_file,'r').readlines()]
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
        os.system('rm '+tmp_file)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=id
    print('training on gpu '+os.environ['CUDA_VISIBLE_DEVICES'])
