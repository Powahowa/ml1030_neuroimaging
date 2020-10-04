#!/bin/bash --login
SBATCH --ntasks 1
SBATCH --tasks-per-node=1
SBATCH --output=hello-world.out
SBATCH --qos debug
SBATCH --time=00:01:00
echo "$(hostname --fqdn): Hellooooooo world!"
