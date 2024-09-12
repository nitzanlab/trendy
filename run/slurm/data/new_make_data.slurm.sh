#!/bin/bash

#SBATCH --job-name=generate-pde-data
#SBATCH --array=1-100%30
#SBATCH --time=00:45:00
#SBATCH -c1
#SBATCH --mem=10g
#SBATCH --output=./logs/slurm/output-%A_%a.out

# Run the Python script for a subset of data
python ./scripts/data/new_make_data.py --task_id $SLURM_ARRAY_TASK_ID --num_samples 4000 --chunks 100 --pde_class BrusselatorFull --output_dir ./data/brusselator/patches/eps_0.5 --measurement_type scattering --save_pde_solutions --epsilon 0.5 --noise_type patches;

# Check if this is the last job in the array (SLURM_ARRAY_TASK_ID is max)
#if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
#    echo "Indexing data"
#    # Only the last job runs the renaming script after all jobs complete
#    python ./scripts/data/sort_data.py --data_dir ./data/NEW_scattering_brusselator_full
#fi
