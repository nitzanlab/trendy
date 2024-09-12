#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/params.py --data_dir ./data/brusselator/no_noise/eps_0.5/

