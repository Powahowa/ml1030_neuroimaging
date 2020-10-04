#!/bin/sh

#SBATCH --job-name=modelling-LSTM-rawvoxels
#SBATCH --output modelling-LSTM-rawvoxels.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling-LSTM-rawvoxels.py