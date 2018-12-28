#! /bin/bash

for i in $(seq 6 10)
do 
	python main.py -logdir=log/10fold -softmax=global -adj_norm=diag -data=COLLAB -dnorm=1 -fold=$i  
	python main.py -logdir=log/10fold -softmax=neibor -adj_norm=diag -data=COLLAB -dnorm=1 -fold=$i  
done

