import argparse 
import os 
parser=argparse.ArgumentParser()
parser.add_argument('-f',default='',type=str)
args=parser.parse_args()
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
