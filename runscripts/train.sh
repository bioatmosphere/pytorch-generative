#!/bin/bash
#SBATCH --job-name=gem
#SBATCH -A m2467
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

cd ../
python train.py --model image_gpt --gpus 1

