#!/bin/bash
#SBATCH --job-name=mnist_train
#SBATCH --output=logs/mnist_lr%j.out
#SBATCH --error=logs/mnist_lr%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:20:00

source ~/.bashrc
module load cuda  # or your cluster's module system
conda activate gpu-env  # activate your Python env

# Set your desired parameters here
LR=0.005
BS=64
EPOCHS=5

python train.py --lr $LR --batch_size $BS --epochs $EPOCHS
