#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c8
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu:1

python ./scripts/tests/fixed_points.py --model_dir ./models/good_models/brusselator/no_noise/eps_0.0/pca_2 --data_dir ./data/brusselator/no_noise/eps_0.0 --save_dir ./data/steady_states/brusselator/pca_2;
