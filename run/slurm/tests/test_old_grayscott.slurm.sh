#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test_old_grayscott.py --model_dir ./models/trendy/run_3 --model_dir2 ./models/good_models/gs_scattering_n_2_to_16/16;
