#!/bin/bash --login
#SBATCH --ntasks 1
#SBATCH --tasks-per-node=1
#SBATCH --output email-test.out
#SBATCH --qos debug
#SBATCH --time=00:01:00
#SBATCH --mail-type=END
#SBATCH --mail-user=TonyMing.Lee@camh.ca
echo "$(hostname --fqdn): Hellooooooo world!"
