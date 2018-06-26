#!/bin/bash -l
#
#SBATCH --job-name=dgc
#SBATCH --time=00:15:00
#SBATCH --constraint=gpu
#SBATCH --output=./slurm_bwd.%j.log

##SBATCH --nodes=32

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.7.0-CrayGNU-17.12-cuda-8.0-python3

# load virtualenv
export WORKON_HOME=~/virtualenv3
source $WORKON_HOME/pytorch-horovod/bin/activate
# mode 0 - exp model
# mode 1 - thd model
# mode 2 - chunk model
# mode 3 - topk model

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES python3 ./test_bwd.py

deactivate
