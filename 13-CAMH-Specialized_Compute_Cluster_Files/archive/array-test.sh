#!/bin/bash --login
SBATCH --array 0-9
SBATCH --ntasks 1
SBATCH --output array-job.out
SBATCH --open-mode append
SBATCH --qos debug
SBATCH --time=00:05:00
echo "$(hostname --fqdn): index ${SLURM_ARRAY_TASK_ID}"
