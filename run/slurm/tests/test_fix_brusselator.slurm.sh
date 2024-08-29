#!/bin/bash
#SBATCH --mem=250m
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test_fix_brusselator.py;
