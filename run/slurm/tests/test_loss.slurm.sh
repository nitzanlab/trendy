#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test_loss.py; 
