#! /bin/bash

for i in $(seq $1 $2)
do
python3 main1.py -fold=$i -model=gcn -pool=mean -gcn_layers=3 -logdir=log/gcn_meanpool_coru -gpu=$3 -data=NCI1
python3 main1.py -fold=$i -model=gcn -pool=mean -gcn_layers=3 -logdir=log/gcn_meanpool_coru -gpu=$3 -data=PROTEINS
python3 main1.py -fold=$i -model=gcn -pool=mean -gcn_layers=3 -logdir=log/gcn_meanpool_coru -gpu=$3 -data=DD
python3 main1.py -fold=$i -model=gcn -pool=mean -gcn_layers=3 -logdir=log/gcn_meanpool_coru -gpu=$3 -data=COLLAB
python3 main1.py -fold=$i -model=gcn -pool=mean -gcn_layers=3 -logdir=log/gcn_meanpool_coru -gpu=$3 -data=REDDIT-BINARY
done
