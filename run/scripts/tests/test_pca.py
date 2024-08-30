from trendy.data import *
from trendy.models import TRENDy, IntegrationScheduler
from trendy.train import *
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pickle
import numpy as np
import os
import time
import json
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='./logs/tb')
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--model_dir', type=str, default='./models/nodes')
parser.add_argument('--non_autonomous', action='store_true')
parser.add_argument('--measurement_type', type=str, default='scattering')
parser.add_argument('--in_shape', nargs='+', default = [2, 64, 64])
parser.add_argument('--num_params', type=int, default=4)
parser.add_argument('--node_hidden_layers', nargs='+', type=int, default=[64,64,64,64])
parser.add_argument('--node_activations', type=str, default='relu')
parser.add_argument('--use_pca', action="store_true")
parser.add_argument('--pretrained', action="store_true")
parser.add_argument('--run_name', type=str, default=None, help='If resuming training with or without optimizer, where are the model and optimizer stored?')
parser.add_argument('--pca_components', type=int, default=2)
parser.add_argument('--log_scale', action='store_true')
parser.add_argument('--num_epochs', type=int,default=10)
parser.add_argument('--batch_size', type=int,default=64)
parser.add_argument('--checkpoint_period', type=int,default=10)
parser.add_argument('--lr', type=float,default=1e-4)
parser.add_argument('--scheduler_type', type=str, default=None)
parser.add_argument('--min_prop', type=float, default=.1)
parser.add_argument('--stop_epoch', type=int, default=None)
parser.add_argument('--loss_order', type=int,default=1)
parser.add_argument('--der_weight', type=float,default=0.0)
parser.add_argument('--burn_in_size', type=float,default=0.0)
parser.add_argument('--dt_est', type=float,default=1e-2)
parser.add_argument('--dt_true', type=float,default=1e-2)
parser.add_argument('--T_est', type=float,default=1.0)
parser.add_argument('--seed', type=int,default=0)
args = parser.parse_args()

# Set seeds for reproducibility
set_seed(args.seed)

# Inititalizing training envrionment
args, device, checkpoint_dir, writer, num_gpus, num_cpus = initialize_training_environment(args)

# Save training arguments for reproducibility
args_dict = vars(args)
with open(os.path.join(checkpoint_dir, 'training_manifest.json'), 'w') as json_file:
    json.dump(args_dict, json_file, indent=4)

# Data loaders
train_ds = SP2VDataset(data_dir=os.path.join(args.data_dir, 'train'))
train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_cpus, shuffle=True, drop_last=True)
test_ds = SP2VDataset(data_dir=os.path.join(args.data_dir, 'test'))
test_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_cpus)

# Set up model
pca_dir = os.path.join('./models/pca', args.data_dir.split('/')[-1] + f'_pca_{args.pca_components}')
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_log_scale=args.log_scale,  use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous, pca_dir=pca_dir)

print(f'Using architecture: {model}')

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# If pretrained, load it, else make sure to fit pca
#model.fit_pca(train_dl)
load_checkpoint(model, args.model_dir, pca_dir=pca_dir, only_load_pca=True)
