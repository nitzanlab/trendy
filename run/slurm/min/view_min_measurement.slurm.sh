#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c8
#SBATCH --time=0:30:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/min/view_min_measurement.py --data_dir ./data/min/train --index 5 --log_scale --resize_shape 128 --pca_components 4 --batch_size 10;
