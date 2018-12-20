#! /bin/bash

python main.py -model=gcn -bsize=20 -gcn_norm=1 -data=REDDIT-MULTI-12K
python main.py -model=gcn -bsize=20 -gcn_norm=0 -data=REDDIT-MULTI-12K
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=COLLAB
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=PROTEINS
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=DD
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=ENZYMES
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=MUTAG
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=PTC
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=IMDBBINARY
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=IMDBMULTI
#python main.py -model=gcn -hidden_dim=128 -bsize=20 -gcn_norm=1 -data=Synthie


#python main.py -model=gcn -data=PROTEINS -gcn_layers=2 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=3 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=4 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=5 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=6 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=9 
#python main.py -model=gcn -data=PROTEINS -gcn_layers=10
#python main.py -model=gcn -data=PROTEINS -gcn_layers=12
#python main.py -model=gcn -data=COLLAB -gcn_layers=2 
#python main.py -model=gcn -data=COLLAB -gcn_layers=3 
#python main.py -model=gcn -data=COLLAB -gcn_layers=4 
#python main.py -model=gcn -data=COLLAB -gcn_layers=5 
#python main.py -model=gcn -data=COLLAB -gcn_layers=6 
#python main.py -model=gcn -data=COLLAB -gcn_layers=9 
#python main.py -model=gcn -data=COLLAB -gcn_layers=10 
#python main.py -model=gcn -data=COLLAB -gcn_layers=12 

