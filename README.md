## This zip file contains source code and two datasets for our paper.

## To run our code, please follow the steps below:

### Dependencies:

Pytorch >=1.0.0, Python >=3.5

### Usage:

* run `pip3 install -r requirement.txt`

* Due to size limit, we provide two dataset for testing, namely the `NCI1` and `PROTEINS` datasets. 

* We provide 6 shell scripts, for training baseline , AttPool-G and AttPool-L models with 10- fold cross validation on NCI1 and PROTEINS datasets, respectively. For example, to train AttPool-G on the NCI1 dataset, please  run `./run_attpool_global_nci1.sh`.
