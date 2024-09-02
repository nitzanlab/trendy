#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c1
#SBATCH --time=00:10:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/tests/test_savemodel.py; 
