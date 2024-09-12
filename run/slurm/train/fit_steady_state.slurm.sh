#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/train/fit_steady_state.py --data_dir ./data;
