#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=agcn -softmax=global -adj_norm=none -relu=lrelu -arch=2 -num_layers=4 -gcn_layers=3 -logdir=log/global_nci1 -data=NCI1
done

rm log/global_nci1/acc_results.txt

python plot1.py -f log/global_nci1 -fold 10
