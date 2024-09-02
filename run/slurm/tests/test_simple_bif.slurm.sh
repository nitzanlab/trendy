#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/test_simple_bif.py --model_dir ./models/tentative/run_2 --data_dir ./data/scattering_brusselator --use_pca; 
