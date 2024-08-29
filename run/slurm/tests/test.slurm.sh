#!/bin/bash
#SBATCH --mem=1m
#SBATCH -c1
#SBATCH --time=00:01:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test.py;
