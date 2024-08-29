#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test_single_fit.py; 
