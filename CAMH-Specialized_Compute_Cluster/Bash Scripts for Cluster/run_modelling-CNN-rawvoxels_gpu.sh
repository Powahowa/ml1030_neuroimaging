#!/bin/sh

#SBATCH --job-name=gpu_modelling-CNN-rawvoxels
#SBATCH --output gpu_modelling-CNN-rawvoxels.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python gpu_test.py