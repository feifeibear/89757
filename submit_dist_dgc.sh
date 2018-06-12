#export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH
#source activate dgc

export BATCH_SIZE=64
export USE_PRUNING=use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=no_use_warmup
export USE_SYNC=no_use_sync
#export MODEL_NAME=alexnet
export MODEL_NAME=vgg16
#export MODEL_NAME=cifar10_shallow
#export MODEL_NAME=resnet
#export MODEL_NAME=widenet
#export MODEL_NAME=mobilenetv2 #resnet
#export RESNET_DEPTH=50
export RESNET_DEPTH=44
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.9
export WEIGHTDECAY=0.0001
export USE_NES=use_nesterov
#export DATASET=imagenet #cifar10
export DATASET=cifar10
export USE_CUDA=cuda.
export PRUNE_MODE=3
export NUM_NODE=2
export USE_CLUSTER=use_cluster

# mode 0 - exp model
# mode 1 - thd model
# mode 2 - chunk model
# mode 3 - topk model

for PRUNE_MODE in 3;
do
mpirun -np ${NUM_NODE} python3 ./main_dist_dgc.py \
  --dataset ${DATASET} \
  --${USE_CLUSTER} \
  --gpus 0 \
  --type torch.${USE_CUDA}FloatTensor \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --print-freq 10 \
  --weight-decay=${WEIGHTDECAY} \
  --pruning_mode=${PRUNE_MODE} \
  --${USE_RESIDUE_ACC} \
  --${USE_NES} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save dist_${NUM_NODE}_${DATASET}_${MODEL_NAME}${RESNET_DEPTH}_${BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}_${USE_NES}_${USE_CUDA}_${PRUNE_MODE}
done


#source deactivate
