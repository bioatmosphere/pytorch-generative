#!/bin/bash
#SBATCH --job-name=gem
#SBATCH -A m2467
#SBATCH -C gpu
#SBATCH -q shared #regular
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load python
conda activate genai
export logdir=/pscratch/sd/w/wangb/pytorch_genai/run
cd ../
python train.py --model vae --epochs 457 --batch-size 128 --logdir $logdir --gpus 1

