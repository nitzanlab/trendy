#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c8
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/get_frequencies.py;
