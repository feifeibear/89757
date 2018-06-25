This code support training CNN with GC for cifar10 and Imagenet dataset

##How to run this code
### Train Cifar10
if you computer do not connect to the internet
1. download cifar10 to ./dataset/pytorch
mkdir -p ./dataset/pytorch/cifar10
download the follows data to that directory
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

2. Modify the submit scripts

run on GPU cluster, where each node has one card
export USE_CLUSTER=use_cluster
set the number of node to run
export NUM_NODE=2
use bsub instead of mpirun

3. run code
To test GC, you should run
./submit_dist_dgc.sh
To test SGD, you should run
./submit_dist_sgd.sh

