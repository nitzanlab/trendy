#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu:1

python ./scripts/eval/view_prediction.py --model_dir ./models/tentative/run_0 --index 1;
