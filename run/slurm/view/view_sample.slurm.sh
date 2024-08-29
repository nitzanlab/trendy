#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/view_sample.py --data_dir ./data/scattering_min/train --index 0;
