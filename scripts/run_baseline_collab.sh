#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=gcn -num_layers=3 -gcn_layers=3 -logdir=log/gcn_collab -data=COLLAB
done

rm log/gcn_collab/acc_results.txt

python plot1.py -f log/gcn_collab -fold 10
