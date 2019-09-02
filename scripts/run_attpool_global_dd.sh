#! /bin/bash
for i in $(seq 1 10)
do
	echo ${i}
	python3 main.py -fold=$i -model=agcn -softmax=global -adj_norm=none -relu=lrelu -arch=2 -num_layers=3 -gcn_layers=3 -logdir=log/global_dd -data=DD
done

rm log/global_dd/acc_results.txt

python plot1.py -f log/global_dd -fold 10
