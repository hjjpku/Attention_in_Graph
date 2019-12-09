## This zip file contains source code and datasets for our ICCV19 paper “AttPool : Towards Hierarchical Feature Representation in Graph Convolutional Networks via Attention Mechanism”

## To run our code, please follow the steps below:

### Dependencies:

Pytorch >=1.0.0, Python >=3.5

### Usage:

* run `pip3 install -r requirement.txt`

* We provide all datasets that have been mentioned in the paper for testing. 

* We provide shell scripts, for training baseline , AttPool-G and AttPool-L models with 10- fold cross validation on datasets, respectively. For example, to train AttPool-G on the NCI1 dataset, please  run `./run_attpool_global_nci1.sh`.

## If you find our work useful, please consider citing:
 
```
@inproceedings{huang2019attpool,
	  title={AttPool: Towards Hierarchical Feature Representation in Graph Convolutional Networks via Attention Mechanism},
	    author={Huang, Jingjia and Li, Zhangheng and Li, Nannan and Liu, Shan and Li, Ge},
		  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
		    pages={6480--6489},
			  year={2019}
}
```
