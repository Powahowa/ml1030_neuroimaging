#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --job-name=modelling_alex
#SBATCH --output modelling_alex.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=06:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling_alex.py