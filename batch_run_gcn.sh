#! /bin/bash

python main.py -model=gcn -gcn_layers=2 
python main.py -model=gcn -gcn_layers=3 
python main.py -model=gcn -gcn_layers=4 
python main.py -model=gcn -gcn_layers=5 
python main.py -model=gcn -gcn_layers=6 
python main.py -model=gcn -gcn_layers=9 
python main.py -model=gcn -gcn_layers=10
python main.py -model=gcn -gcn_layers=12

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

