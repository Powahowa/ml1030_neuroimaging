#!/bin/sh
#SBATCH --job-name=modelling-rawvoxels-SCCmin-SFCM_confoundsOut_20-100slice
#SBATCH --output run_modelling-rawvoxels-SCCmin-SFCM_confoundsOut_20-100slice.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python modelling-rawvoxels-SCCmin-SFCM_confoundsOut_20-100slice.py