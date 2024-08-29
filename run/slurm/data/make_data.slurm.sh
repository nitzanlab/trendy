#!/bin/bash

#SBATCH --job-name=generate-pde-data
#SBATCH --array=1-100%30
#SBATCH --time=01:00:00
#SBATCH -c1
#SBATCH --mem=32g
#SBATCH --output=./logs/slurm/output-%A_%a.out

# Run the Python script for a subset of data
python ./scripts/data/make_data.py --task_id $SLURM_ARRAY_TASK_ID --chunks 100 --pde_class GrayScott --num_bins 50 --output_dir ./data/scattering_grayscott_focused --measurement_type scattering --save_pde_solutions;
