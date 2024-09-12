#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c8
#SBATCH --time=06:00:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/train/train_model.py --data_dir ./data/scattering_min --model_dir ./models/tentative --log_dir ./logs/tb --num_epochs 2000 --batch_size 64 --der_weight 0.0 --dt_est 1e-5 --T_est 5e-2  --dt_true .2512 --use_pca --pca_components 2 --num_params 2 --lr 1e-4 --node_hidden_layers 64 64 64 64 --non_autonomous --log_estimate --clip_target 25 --log_scale;


