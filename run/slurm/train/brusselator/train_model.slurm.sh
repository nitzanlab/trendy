#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c8
#SBATCH --time=03:00:00
#SBATCH --output=./logs/slurm/output-%j.out
#SBATCH --gres=gpu,vmem:6G

python ./scripts/train/train_model.py --data_dir ./data/brusselator/patches/eps_0.25 --model_dir ./models/tentative --log_dir ./logs/tb --num_epochs 1000 --batch_size 64 --der_weight 1e-4 --dt_est 1e-2 --T_est 1.0  --dt_true 1e-2 --use_pca --pca_components 8 --num_params 4 --lr 1e-4 --node_hidden_layers 64 64 64 64 --log_estimate --pretrained_weights --run_name run_5;


