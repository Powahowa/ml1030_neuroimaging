#!/bin/sh
#SBATCH --job-name=run_masker_SCC
#SBATCH --output run_maskerApplication_SCC.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128gb
#SBATCH --time=3:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python maskerApplication_SCC.py
