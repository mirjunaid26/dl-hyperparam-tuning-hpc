#!/bin/bash
for lr in 0.001 0.005 0.01; do
  for bs in 32 64; do
    sbatch --export=LR=$lr,BS=$bs,EPOCHS=6 job_train.slurm
  done
done
