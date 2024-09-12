#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c8
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/grayscott_classify.py --data_dir ./data/scattering_grayscott_full --n_components 16 --n_clusters 5 --batch_size 128;
