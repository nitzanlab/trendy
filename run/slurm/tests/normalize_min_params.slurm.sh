#!/bin/bash
#SBATCH --mem=10m
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu:1

python ./scripts/tests/normalize_min_params.py;
