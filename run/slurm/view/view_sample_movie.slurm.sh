#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/view_sample_movie.py --pde_name GrayScott --step 40 --params .054 .062 .1 .05 --noise_type boundaries;

