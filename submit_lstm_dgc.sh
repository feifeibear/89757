export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
source activate dgcv2
#export CUDA_VISIBLE_DEVICES=4,5
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export NUM_NODE=4

mpirun -np ${NUM_NODE} \
  python ./main_lstm.py \
  --use_pruning \
  --cuda \
  --batch_size 5 \
  --emsize 150 \
  --nhid 150 \
  --dropout 0.65 \
  --epochs 10 \
  --tied \
  --save_dir lstm_dgc_${NUM_NODE} #2>&1 | tee org.log
source deactivate
