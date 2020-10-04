#!/bin/sh

#SBATCH --job-name=feature-extraction-alex_SCC
#SBATCH --output feature-extraction-alex_SCC.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=400gb
#SBATCH --time=6:00:00

export XDG_RUNTIME_DIR=/nethome/kcni/$USER/XDG_RUNTIME_DIR
python feature-extraction-alex_SCC.py