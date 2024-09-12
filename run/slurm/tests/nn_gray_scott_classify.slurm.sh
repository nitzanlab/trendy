#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c8
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/nn_grayscott_classify.py --data_dir ./data/scattering_grayscott_full --n_clusters 4 --batch_size 64 --use_pca --pca_components 16 --log_scale;
