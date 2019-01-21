#! /bin/bash

for i in $(seq $1 $2)
do
python3 main.py -fold=$i -softmax=global -adj_norm=none -relu=relu -att_out=1 -pool=sum -logdir=log/att_x_relu -gpu=$3 -data=NCI1
python3 main.py -fold=$i -softmax=global -adj_norm=none -relu=relu -att_out=1 -pool=sum -logdir=log/att_x_relu -gpu=$3 -data=PROTEINS
python3 main.py -fold=$i -softmax=global -adj_norm=none -relu=relu -att_out=1 -pool=sum -logdir=log/att_x_relu -gpu=$3 -data=MUTAG
python3 main.py -fold=$i -softmax=global -adj_norm=none -relu=relu -att_out=1 -pool=sum -logdir=log/att_x_relu -gpu=$3 -data=COLLAB
done
