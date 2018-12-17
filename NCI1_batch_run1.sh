#! /bin/bash
python main.py -softmax=mix -adj_norm=none -khop=2 
python main.py -softmax=mix -adj_norm=none -khop=2 -tau=5
python main.py -softmax=mix -adj_norm=none -khop=3
python main.py -softmax=mix -adj_norm=diag -khop=2
python main.py -softmax=mix -adj_norm=diag -khop=2 -tau=5
python main.py -softmax=mix -adj_norm=diag -khop=3
python main.py -softmax=mix -adj_norm=tanh -khop=2 
python main.py -softmax=mix -adj_norm=tanh -khop=2 -tau=5
python main.py -softmax=mix -adj_norm=tanh -khop=3
