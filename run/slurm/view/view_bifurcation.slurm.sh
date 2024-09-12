#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/view/view_bifurcation.py --model_dir ./models/good_models/brusselator/no_noise/eps_0.0/pca_2 --pde_name Brusselator --base_params 1.5 1.0 .1 .2 --bp_ind 0 --bp_min 1.5 --bp_max 4.0 --num_sols 10 --save_steady_state;
