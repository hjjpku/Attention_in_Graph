import argparse 
import math
import sys
import os 
parser=argparse.ArgumentParser()
parser.add_argument('-f',default='',type=str)
parser.add_argument('-fold',default=0,type=int)
parser.add_argument('-avg',default=1,type=int)
parser.add_argument('-test',default=1,type=int)
args=parser.parse_args()

files=[os.path.join(args.f,x) for x in os.listdir(args.f)]
test_acc={}
train_acc={}
train_cnt={}
test_cnt={}
train_record=[[]]*1000
test_record=[[]]*1000
for x in files:
    f=open(x,'r')
    for l in f.readlines():
        if 'of epoch' not in l:
            continue
        words=l.strip().split(' ')
        epoch=int(words[words.index('epoch')+1][:-1])
        if epoch not in train_acc.keys():
            train_acc[epoch]=0
            test_acc[epoch]=0
            train_cnt[epoch]=0
            test_cnt[epoch]=0
        if args.avg:
            acc=float(words[words.index('avg_acc')+1])
        else:
            acc=float(words[words.index('acc')+1])
        if 'test' in l:
            test_cnt[epoch]+=1
            test_acc[epoch]+=acc
            test_record[epoch].append(acc)
        else:
            train_cnt[epoch]+=1
            train_acc[epoch]+=acc
            train_record[epoch].append(acc)

def var(l,avg):
    sqsum=0
    for x in l:
        sqsum+=(x-avg)**2
    return math.sqrt(sqsum/len(l)) 

max_acc=0
max_epoch=-1
for i,x in enumerate(sorted(train_acc.keys())):
    assert train_cnt[x]==args.fold and test_cnt[x]==args.fold
    if not args.test:
        acc=train_acc[x]/args.fold
        print('%d\t%.4f\t%.6f'%(i,acc,var(train_record[i],acc)))
    else:
        acc=test_acc[x]/args.fold
        print('%d\t%.4f\t%.6f'%(i,acc,var(test_record[i],acc)))
    if max_acc<acc:
        max_acc=acc
        max_epoch=i

print('max: %.4f, epoch %d'%(max_acc,max_epoch))
