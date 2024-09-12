#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c8
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/output-%j.out

python ./scripts/tests/pde_simple_bif.py --data_dir ./data/scattering_brusselator --n_components 2 --batch_size 128;
