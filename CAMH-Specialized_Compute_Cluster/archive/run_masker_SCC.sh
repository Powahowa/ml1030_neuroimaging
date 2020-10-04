#!/bin/sh
#SBATCH --job-name=run_masker_SCC
#SBATCH --output run_masker_SCC.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64gb
#SBATCH --time=1:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python masker_SCC.py
