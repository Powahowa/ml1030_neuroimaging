#!/bin/sh

#SBATCH --job-name=modelling-CNN-rawvoxels2
#SBATCH --output modelling-CNN-rawvoxels2.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling-CNN-rawvoxels2.py
