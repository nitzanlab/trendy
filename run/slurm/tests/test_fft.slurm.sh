#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu:1

python ./scripts/tests/test_fft.py --model_dir ./models/trendy/run_0;
