#!/bin/sh
#SBATCH --job-name=modelling-rawvoxels-sep9-STCM_confoundsOut_43-103slice
#SBATCH --output run_modelling-rawvoxels-sep9-STCM_confoundsOut_43-103slice.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling-rawvoxels-sep9.py