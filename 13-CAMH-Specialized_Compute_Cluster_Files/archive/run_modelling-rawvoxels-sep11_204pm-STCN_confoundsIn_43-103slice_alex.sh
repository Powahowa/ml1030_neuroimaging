#!/bin/sh
#SBATCH --job-name=modelling-rawvoxels-sep11_204pm-STCN_confoundsIn_43-103slice_alex
#SBATCH --output modelling-rawvoxels-sep11_204pm-STCN_confoundsIn_43-103slice_alex.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling_alex.py