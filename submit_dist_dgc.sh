#export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH
source activate dgc

export BATCH_SIZE=64
export USE_PRUNING=use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=no_use_warmup
export USE_SYNC=no_use_sync
#export MODEL_NAME=alexnet
#export MODEL_NAME=vgg16
#export MODEL_NAME=cifar10_shallow
#export MODEL_NAME=resnet
export MODEL_NAME=widenet
#export MODEL_NAME=mobilenetv2 #resnet
#export RESNET_DEPTH=18
export RESNET_DEPTH=44
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.9
export WEIGHTDECAY=0.0001
export USE_NES=use_nesterov
#export DATASET=imagenet #cifar10
export DATASET=cifar10
export NUM_NODE=4
export USE_CUDA=cuda.

mpirun -np ${NUM_NODE} python3 ./main_dist_dgc.py \
  --dataset ${DATASET} \
  --gpus 4,0,1,2 \
  --type torch.${USE_CUDA}FloatTensor \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --print-freq 10 \
  --weight-decay=${WEIGHTDECAY} \
  --pruning_mode=2 \
  --${USE_RESIDUE_ACC} \
  --${USE_NES} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save dist_${NUM_NODE}_${DATASET}_${MODEL_NAME}${RESNET_DEPTH}_${BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}_${USE_NES}_${USE_CUDA}_thd


source deactivate
