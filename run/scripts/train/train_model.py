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
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_log_scale=args.log_scale,  use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous)

print(f'Using architecture: {model}')

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# If pretrained, load it, else make sure to fit pca
if args.pretrained:
    print("Loading pretrained model", flush=True)
    model, opt, start_epoch = load_checkpoint(model, checkpoint_dir, optimizer=opt, device=device)
else:
    model.fit_pca(train_dl, model_dir=checkpoint_dir)
    start_epoch = 0

# TODO: remove!
#train_ds = SP2VDataset(data_dir=os.path.join(args.data_dir, 'train'))
#train_ds = Subset(train_ds, [1])
#train_dl = DataLoader(train_ds, batch_size=1, num_workers=num_cpus, shuffle=False, drop_last=True)
#test_dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=num_cpus)

# Turn off gradients for PCA part
model.pca_layer.linear.weight.requires_grad = False
model.pca_layer.mean.requires_grad = False

# Set device
model = model.to(device)

# Loss
criterion = NthOrderLoss(n=args.loss_order, dt_est = args.dt_est, dt_true = args.dt_true, der_weight = args.der_weight, burn_in_size=args.burn_in_size) 

# Scheduler, if any
max_samples = int(args.T_est // args.dt_est)
scheduler = IntegrationScheduler(args.scheduler_type, max_samples, args.stop_epoch, min_prop=args.min_prop)
print(f'Using {args.scheduler_type} scheduler with max_samples {max_samples} and min_samples {int(args.min_prop*max_samples)}.')

# Parallelize model
model = torch.nn.DataParallel(model)
print(f"Number of GPUs being used: {len(model.device_ids)}")

# Run training loop
view_losses = {'train': 0, 'test': 0}

for epoch in range(start_epoch, start_epoch + args.num_epochs):
    for dl, mode in zip([train_dl, test_dl], ['train','test']):
        start = time.time()

        if scheduler is not None:
            # Update scheduler
            scheduler.update(model, epoch)
            print(scheduler.current_prop)

        loss = run_epoch(dl, model, criterion, opt, scheduler, train=(mode=='train'), use_log_scale=args.log_scale, device=device)
        stop = time.time()
        view_losses[mode] += loss
        model_core = model.module if isinstance(model, torch.nn.DataParallel) else model
        print(f'{mode} epoch: {epoch}. Loss: {loss:.4f}. Time: {stop-start:.4f} s. Current samples: {int(model_core.NODE.T / model_core.NODE.dt)}', flush=True)
        if epoch % args.checkpoint_period == 0 and epoch > start_epoch:
            view_losses[mode]/= args.checkpoint_period
            writer.add_scalar(f'Loss/{mode}', view_losses[mode], epoch)
            if scheduler is not None:
                writer.add_scalar(f'Current prop.', scheduler.current_prop, epoch)
            save_checkpoint(model, opt, epoch, checkpoint_dir)
            view_losses[mode] = 0

save_checkpoint(model, opt, epoch, checkpoint_dir)
print("Training completed.", flush=True)
