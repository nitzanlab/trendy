#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/train/get_steady_state.py --model_dir ./models/good_models/brusselator/no_noise/eps_0.0/pca_4 --pde_name BrusselatorFull --num_samples 100 --save_dir ./data/ss4;
