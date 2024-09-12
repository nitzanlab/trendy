#!/bin/bash

#SBATCH --job-name=generate-pde-data
#SBATCH --array=1-2%2
#SBATCH --time=01:00:00
#SBATCH -c1
#SBATCH --mem=10g
#SBATCH --output=./logs/slurm/output-%A_%a.out

# Run the Python script for a subset of data
python ./scripts/tests/tmp_get_inits.py --task_id $SLURM_ARRAY_TASK_ID --chunks 2 --pde_class BrusselatorFull --num_bins 50 --output_dir ./data/tmp_brusselator_inits --measurement_type scattering --save_pde_solutions;
