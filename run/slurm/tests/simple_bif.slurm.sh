#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c8
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/simple_bif.py --data_dir ./data/scattering_grayscott_full --use_pca --n_components 16;
