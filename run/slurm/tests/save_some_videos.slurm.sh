#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/save_some_videos.py --data_dir ./data/brusselator/no_noise/eps_0.0/train --num_samples 10 --pde_name BrusselatorFull --noise_type boundaries --save_dir ./data/some_sample_movies/boundaries;
