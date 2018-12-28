#! /bin/bash 

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold_gcn -model=gcn -data=ENZYMES -fold=$i  
done

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold_gcn -model=gcn -data=NCI1 -fold=$i  
done

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold_gcn -model=gcn -data=PROTEINS -fold=$i  
done

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold_gcn -model=gcn -data=DD -fold=$i  
done

for i in $(seq 1 10)
do 
	python main.py -logdir=log/10fold_gcn -model=gcn -data=COLLAB  -fold=$i  
done
