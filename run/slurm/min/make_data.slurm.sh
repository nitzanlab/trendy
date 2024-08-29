#!/bin/bash

#SBATCH --job-name=generate-pde-data
#SBATCH --array=1-20%20
#SBATCH --time=01:00:00
#SBATCH -c1
#SBATCH --mem=32g
#SBATCH --output=./logs/slurm/output-%A_%a.out

# Run the Python script for a subset of data
python ./scripts/min/make_data.py --task_id $SLURM_ARRAY_TASK_ID --data_dir ./data/min --output_dir ./data/scattering_min --measurement_type scattering --save_pde_solutions  --chunks 20;
