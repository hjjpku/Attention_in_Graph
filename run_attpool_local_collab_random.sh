#! /bin/bash
for i in $(seq 1 10)
do
	python3 main.py -fold=$i -model=agcn -softmax=neibor -dnorm=1 -adj_norm=none -relu=relu -arch=2 -num_layers=4 -gcn_layers=3 -logdir=log/local_collab_random -data=COLLAB -sort=random_sample
done

rm log/local_collab/acc_results.txt

python plot1.py -f log/local_collab -fold 10
