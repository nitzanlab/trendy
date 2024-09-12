#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/view_sample.py --data_dir ./data/scattering_grayscott_full/train --index 8 --pca_dir ./models/pca/scattering_grayscott_full_pca_16 --pca_components 16 --log_scale --clip_target 1000;

