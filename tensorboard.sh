#!/bin/bash -l

#SBATCH --account=p200140
#SBATCH --job-name=TensorBoard
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread

# Load Modules
export LOGDIR=/project/home/p200140/pDDLES/FNet/FNet128-norm-4-500epochs
module load TensorFlow

# Launch a tensorboard instance
srun tensorboard --logdir $LOGDIR --port 6006 1> script2.out 2> script2.err &
wait
