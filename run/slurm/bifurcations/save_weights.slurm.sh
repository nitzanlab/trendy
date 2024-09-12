#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=00:20:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/bifurcations/save_weights.py --model_super_dir ./models/good_models/brusselator
