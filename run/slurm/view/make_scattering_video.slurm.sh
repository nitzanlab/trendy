#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=0:30:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/view/make_scattering_video.py;
