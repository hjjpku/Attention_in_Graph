#! /bin/bash

for i in $(seq $1 $2)
do
python3 main.py -fold=$i -softmax=neibor -adj_norm=none -dnorm=1 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=NCI1
python3 main.py -fold=$i -softmax=neibor -adj_norm=none -dnorm=1 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=PROTEINS
python3 main.py -fold=$i -softmax=neibor -adj_norm=none -dnorm=1 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=MUTAG
python3 main.py -fold=$i -softmax=neibor -adj_norm=none -dnorm=1 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=COLLAB
#python3 main.py -fold=$i -softmax=neibor -adj_norm=diag -dnorm=1 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=COLLAB
#python3 main.py -fold=$i -softmax=neibor -adj_norm=diag -dnorm=1 -num_layers=4 -gcn_layers=8 -relu=lrelu -att_out=1 -pool=sum -logdir=log/att_neibor_lrelu -gpu=$3 -data=REDDIT-BINARY
done
