#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=0:10:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/view_scattering_filter.py;
