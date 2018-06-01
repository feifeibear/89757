#export PATH=/home/fangjr/miniconda3/lib:/home/fangjr/miniconda3/bin:$PATH

export BATCH_SIZE=64
export MINI_BATCH_SIZE=64
export USE_PRUNING=use_pruning
export USE_RESIDUE_ACC=use_residue_acc
export USE_WARMUP=no_use_warmup
export USE_SYNC=no_use_sync
export MODEL_NAME=resnet
#export MODEL_NAME=mobilenetv2 #resnet
export RESNET_DEPTH=44
#export RESNET_DEPTH=18
export LRSCALE=lr_bb_fix
# a Vanilla SGD, useless because py in models decides mom
export MOMENTUM=0.9
export WEIGHTDECAY=0.0001
export USE_NES=use_nesterov
#export DATASET=imagenet
export DATASET=cifar10

python3 main_dgc.py \
  --gpus 1,0,3,4 \
  --type torch.cuda.FloatTensor \
  --dataset ${DATASET} \
  --resnet_depth=${RESNET_DEPTH} \
  --model ${MODEL_NAME} \
  --epochs 164 \
  --b ${BATCH_SIZE} \
  --mini-batch-size ${MINI_BATCH_SIZE} \
  --momentum=${MOMENTUM} \
  --weight-decay=${WEIGHTDECAY} \
  --${USE_RESIDUE_ACC} \
  --${USE_NES} \
  --${USE_PRUNING} \
  --${USE_WARMUP} \
  --${USE_SYNC} \
  --${LRSCALE} \
  --save dgc_${DATASET}_${MODEL_NAME}${RESNET_DEPTH}_${BATCH_SIZE}_${MINI_BATCH_SIZE}_${USE_PRUNING}_${USE_RESIDUE_ACC}_${USE_WARMUP}_${USE_SYNC}_${USE_NES}_fast

#--use_pruning ${USE_PRUNING}
#--use_residue_acc ${USE_RESIDUE_ACC}
#--resume ./TrainingResults/momentum_cifar10_resnet_128_64_use_pruning_use_residue_acc_use_warmup_no_use_sync/checkpoint.pth.tar
