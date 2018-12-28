#! /bin/bash


#bashfor i in $(seq 1 10)
#bashdo 
#bash	python main.py -logdir=log/10fold -softmax=global -adj_norm=diag -dnorm=1 -data=PROTEINS -fold=$i  
#bash	python main.py -logdir=log/10fold -softmax=neibor -adj_norm=diag -dnorm=1 -data=PROTEINS -fold=$i  
#bashdone

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold -softmax=neibor -adj_norm=diag -dnorm=1 -data=DD -fold=$i  
done


