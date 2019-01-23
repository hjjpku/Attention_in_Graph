#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=gcn -gcn_layers=3 -logdir=log/gcn_nci1 -data=NCI1 
	python3 main.py -fold=$i -model=gcn -gcn_layers=3 -logdir=log/gcn_proteins -data=PROTEINS
done

python3 sort.py -f log/gcn_nci1/acc_results.txt -sort=1 -fold=10
python3 sort.py -f log/gcn_proteins/acc_results.txt -sort=1 -fold=10

echo GCN baseline 10-fold average accuracy on NCI1 dataset: 
cat log/gcn_nci1/cal_results.txt

echo GCN baseline 10-fold average accuracy on PROTEINS dataset: 
cat log/gcn_proteins/cal_results.txt
