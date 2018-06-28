export MODEL_NAME=resnet
export PRUNE_MODE=3
export USE_PRUNING=use_pruning
export NUM_NODE=2

sh ./sw_submit_benchmark.sh

for USE_PRUNING in use_pruning no_use_pruning; do
for NUM_NODE in 1 4 8 16 32; do
for MODEL_NAME in resnet alexnet; do
for PRUNE_MODE in 3 7; do
  sh ./sw_submit_benchmark.sh
done
done
done
done
