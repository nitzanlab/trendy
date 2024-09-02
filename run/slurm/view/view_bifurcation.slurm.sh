#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c1
#SBATCH --time=0:20:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/view/view_bifurcation.py --model_dir ./models/tentative/run_2 --pde_name Brusselator --base_params 1.5 1.0 .1 .2 --bp_ind 1 --bp_min 1. --bp_max 4. --num_sols 100 --view_estimate
