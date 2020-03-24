#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=agcn -softmax=neibor -dnorm=1 -adj_norm=none -relu=relu -arch=2 -num_layers=4 -gcn_layers=3 -logdir=log/local_nci1_sample -data=NCI1 -sort=sample
done

rm log/local_nci1/acc_results.txt

python plot1.py -f log/local_nci1 -fold 10
