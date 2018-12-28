#! /bin/bash


for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold -softmax=global -adj_norm=diag -dnorm=1 -data=PROTEINS -fold=$i  
	python main.py -logdir=log/10fold -softmax=neibor -adj_norm=diag -dnorm=1 -data=PROTEINS -fold=$i  
done

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold -softmax=global -adj_norm=diag -dnorm=1 -data=DD -fold=$i  
	python main.py -logdir=log/10fold -softmax=neibor -adj_norm=diag -dnorm=1 -data=DD -fold=$i  
done


