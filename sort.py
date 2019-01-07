import argparse 
import sys
import os 
parser=argparse.ArgumentParser()
parser.add_argument('-f',default='',type=str)
parser.add_argument('-sort',default=1,type=int)
parser.add_argument('-fold',default=0,type=int)

args=parser.parse_args()
if args.sort:
	r=open(args.f,'r')
	w=open(args.f+'_tmp','w')
	dic={}
	for l in r.readlines():
		s=l.strip().split(' ')[0]
		dic[s]=l

	for x in sorted(dic):
		w.write(dic[x])
	r.close()
	w.close()
	os.system('mv '+args.f+'_tmp'+' '+args.f)

cnt={}
folds={}
dic_acc={}
dic_avg_acc={}
if args.fold>0:
	r=open(args.f,'r')
	w=open(args.f[0:args.f.rfind('/')+1]+'cal_results.txt','w')
	sys.stdout=w
	for l in r.readlines():
		s,acc,avg_acc=l.strip().split(' ')
		tmp=s.split('_')
		new_s='_'.join([x for x in tmp if 'fold' not in x])
		acc=float(acc[0:acc.find('(')])
		avg_acc=float(avg_acc[0:avg_acc.find('(')])
		fold=[x for x in tmp if 'fold' in x]
		if new_s not in cnt.keys():
			cnt[new_s]=0
			folds[new_s]=[]
			dic_acc[new_s]=0
			dic_avg_acc[new_s]=0
		cnt[new_s]+=1
		folds[new_s]+=fold
		dic_acc[new_s]+=acc
		dic_avg_acc[new_s]+=avg_acc
	for x in cnt:
		dic_acc[x]=dic_acc[x]/cnt[x]
		dic_avg_acc[x]=dic_avg_acc[x]/cnt[x]
		folds[x]=','.join(folds[x])
		print(x,' acc=%.4f avg_acc=%.4f'%(dic_acc[x],dic_avg_acc[x]))
		print('%d folds:'%cnt[x],folds[x])

	w.close()
