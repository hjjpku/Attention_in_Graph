#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=gcn -num_layers=3 -gcn_layers=3 -logdir=log/gcn_proteins -data=PROTEINS
done

rm log/gcn_proteins/acc_results.txt

python plot1.py -f log/gcn_proteins -fold 10
