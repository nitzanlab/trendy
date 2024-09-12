#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c8
#SBATCH --time=0:30:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/new_pca.py --data_dir ./data/scattering_min_safe/ --n_components 4;
