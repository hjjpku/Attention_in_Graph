#! /bin/bash

#python main.py -model=gcn -num_layers=3 >>gcn_3layer.txt
##python main.py -model=gcn -num_layers=6 -learning_rate=1e-3 >>gcn_6layer_1e-3.txt
#python main.py -model=agcn -softmax=neibor -tau=1 -percent=0.5 >>agcn_tau1_k0.5.txt
#python main.py -model=agcn -softmax=neibor -tau=5 -percent=0.5 >>agcn_tau5_k0.5.txt
#python main.py -model=agcn -softmax=neibor -tau=5 -percent=0.3 >>agcn_tau5_k0.3.txt
#python main.py -model=agcn -softmax=neibor -tau=5 -percent=0.2 >>agcn_tau5_k0.2.txt
#python main.py -model=agcn -softmax=neibor -tau=5 -percent=0.75 >>agcn_tau5_k0.75.txt
#python main.py -model=agcn -softmax=neibor -tau=10 -percent=0.5 >>agcn_tau10_k0.5.txt
#python main.py -model=agcn -softmax=neibor -tau=20 -percent=0.5 >>agcn_tau20_k0.5.txt

python main.py -softmax=neibor -adj_norm=none -khop=2
python main.py -softmax=neibor -adj_norm=none -khop=2 -tau=5
python main.py -softmax=neibor -adj_norm=none -khop=3
python main.py -softmax=neibor -adj_norm=diag -khop=2 
python main.py -softmax=neibor -adj_norm=diag -khop=2 -tau=5 
python main.py -softmax=neibor -adj_norm=diag -khop=3
python main.py -softmax=neibor -adj_norm=tanh -khop=2
python main.py -softmax=neibor -adj_norm=tanh -khop=2 -tau=5
python main.py -softmax=neibor -adj_norm=tanh -khop=3

#python main.py -softmax=neibor -adj_norm=none 
#python main.py -softmax=neibor -adj_norm=diag 
#python main.py -softmax=neibor -adj_norm=tanh 
#python main.py -softmax=neibor -adj_norm=mix 
#python main.py -softmax=mix -adj_norm=none 
#python main.py -softmax=mix -adj_norm=diag 
#python main.py -softmax=mix -adj_norm=tanh 
#python main.py -softmax=mix -adj_norm=mix 

#
#
#python main.py -model=agcn -learning_rate=1e-1 >>agcn_1e-1.txt
#python main.py -model=agcn -learning_rate=1e-2 >>agcn_1e-2.txt
#python main.py -model=agcn -learning_rate=1e-3 >>agcn_1e-3.txt
#python main.py -model=agcn -learning_rate=1e-4 >>agcn_1e-4.txt
#python main.py -model=agcn -learning_rate=1e-5 >>agcn_1e-5.txt
#python main.py -model=agcn -learning_rate=1e-6 >>agcn_1e-6.txt
#
#python main.py -model=agcn -learning_rate=1e-4 -feat_mode=raw >>agcn_1e-4_raw.txt
#python main.py -model=agcn -learning_rate=1e-4 -gcn_layers=4 >>agcn_1e-4_4_layer.txt
#python main.py -model=agcn -learning_rate=1e-4 -gcn_norm=1 >>agcn_1e-4_norm.txt
#python main.py -model=agcn -learning_rate=1e-4 -relu=0 >>agcn_1e-4_no_relu.txt
#python main.py -model=agcn -learning_rate=1e-4 -mlp_layers=1 >>agcn_1e-4_1_mlp.txt
#python main.py -model=agcn -learning_rate=1e-4 -pool=max >>agcn_1e-4_max_pool.txt
#python main.py -model=agcn -learning_rate=1e-4 -hidden_dim=128 >>agcn_1e-4_128_hidden.txt
#
