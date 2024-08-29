#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/view_sample_movie.py --pde_name Min --step 1 --data_dir ./data/min/train --index 1;
