#! /bin/bash

python main.py   -softmax=global -adj_norm=none -data=NCI1 -wb_init=normal 

python main.py   -softmax=neibor -adj_norm=none -data=NCI1 -wb_init=normal

python main.py   -softmax=global -adj_norm=none -data=PROTEINS -wb_init=normal 

python main.py   -softmax=neibor -adj_norm=none -data=PROTEINS -wb_init=normal

python main.py   -softmax=global -adj_norm=none -data=COLLAB -wb_init=normal 

python main.py   -softmax=neibor -adj_norm=none -data=COLLAB -wb_init=normal

