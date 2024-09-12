#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/data/trim_data.py --data_dir ./data/brusselator/patches/eps_0.25/

