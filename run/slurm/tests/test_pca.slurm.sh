#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c8
#SBATCH --time=00:30:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu:1

python ./scripts/tests/test_pca.py --data_dir ./data/scattering_brusselator --model_dir ./models/tentative --log_dir ./logs/tb --num_epochs 0 --batch_size 64 --der_weight 0.0 --dt_est .0087 --T_est 45.0  --dt_true 1e-3 --use_pca --pca_components 16 --num_params 4 --lr 1e-4 --node_hidden_layers 100 --scheduler_type linear --stop_epoch 15000 --min_prop .07 --non_autonomous;


