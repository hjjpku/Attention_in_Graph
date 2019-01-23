#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=agcn -softmax=global -adj_norm=none -att_out=1 -pool=sum -num_layers=3 -gcn_layers=6 -logdir=log/attpool_g_nci1 -data=NCI1 
	python3 main.py -fold=$i -model=agcn -softmax=global -adj_norm=none -att_out=1 -pool=sum -num_layers=3 -gcn_layers=6 -logdir=log/attpool_g_proteins -data=PROTEINS
done

python3 sort.py -f log/attpool_g_nci1/acc_results.txt -sort=1 -fold=10
python3 sort.py -f log/attpool_g_proteins/acc_results.txt -sort=1 -fold=10

echo AttPool-global 10-fold average accuracy on NCI1 dataset: 
cat log/attpool_g_nci1/cal_results.txt

echo AttPool-global 10-fold average accuracy on PROTEINS dataset: 
cat log/attpool_g_proteins/cal_results.txt
