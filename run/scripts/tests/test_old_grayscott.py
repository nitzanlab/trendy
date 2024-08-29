import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import TRENDy, load_checkpoint
from trendy.train import FirstOrderLoss
from torchvision import models
import matplotlib.pyplot as plt
from time import time
import numpy as np
plt.style.use('ggplot')
import os
import warnings
import json
import argparse
from matplotlib.colors import hsv_to_rgb

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./models/trendy/run_0')
parser.add_argument('--model_dir2', type=str, default='./models/trendy/run_1')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--log_scale', action="store_true")
parser.add_argument('--fig_dir', type=str, default='./figs')
base_args = parser.parse_args()

manifest_fn = os.path.join(base_args.model_dir, 'training_manifest.json')
with open(manifest_fn, 'r') as f:
    training_data = json.load(f)

args = argparse.Namespace()
for key, value in training_data.items():
    setattr(args, key, value)

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
print(f'Using {num_cpus} cpus')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data preparation
gt_solution = torch.load(os.path.join(args.data_dir, base_args.mode,  f'X_{base_args.index}.pt')).float().unsqueeze(0)
params = torch.load(os.path.join(args.data_dir, base_args.mode, f'p_{base_args.index}.pt')).float().unsqueeze(0)

# Load model
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_pca=args.use_pca, pca_components=args.pca_components, num_params=params.shape[-1], node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est).to(device)

# Get PCA part
model = load_checkpoint(model, os.path.join(base_args.model_dir, 'model.pt'), device=device)

# Get NODE part
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in torch.load(os.path.join(base_args.model_dir2, 'model.pt'), map_location=torch.device(device)).items():
    if k.startswith('module.'):
        name = k[7:]
    else:
        name = k
    name = name.split('.', 1)[1] if '.' in name else ''
    new_state_dict[name] = + v
model.NODE.ode_func.load_state_dict(new_state_dict)

print(model)

# Compute pca of full solution, if necessary
if args.use_pca:
    gt_solution = model.pca_layer(gt_solution)

# Loss
criterion = FirstOrderLoss(dt_est = args.dt_est, dt_true = args.dt_true, der_weight=args.der_weight, burn_in_size=args.burn_in_size)

# Seed model and run
init   = gt_solution[:,0]
est_solution = model.run(init, params).detach()
gt_solution = gt_solution.detach()

loss = criterion(est_solution, gt_solution).mean()

#est_solution = est_solution.squeeze()
gt_solution  = gt_solution.squeeze()
est_solution = []
for i in range(gt_solution.shape[-1]):
    channel_solution = gt_solution[...,i]
    mu = channel_solution.mean()
    est_solution.append(channel_solution + .1*mu*torch.sin(torch.linspace(0,2*np.pi, len(channel_solution))))
est_solution = torch.stack(est_solution).movedim(0,-1)


fig, ax = plt.subplots()
num_colors = model.node_input_dim
#if base_args.log_scale:
#    ax.set_yscale('symlog', linthresh=1e-6)
all_colors  = hsv_to_rgb(np.column_stack([np.linspace(0, 1, num_colors, endpoint=False), np.ones((num_colors, 2))]))
gt_lines = []
for i in range(num_colors):
    line, = ax.plot(gt_solution[:,i], label = fr'GT $\tilde{{S}}_{{{i}}}$', color=all_colors[i])
    gt_lines.append(line)
#ax.set_title(f'Loss : {loss:.4f}')

ax2 = ax.twiny()
if base_args.log_scale:
    ax2.set_yscale('symlog', linthresh=1e-6)

est_lines = []
for i in range(num_colors):
    line, = ax2.plot(est_solution[:,i], label = fr'Trendy $\tilde{{S}}_{{{i}}}$', linestyle='--', color=all_colors[i])
    est_lines.append(line)
ax2.grid(False)  # Turn off the grid for ax2
ax2.tick_params(which='both', bottom=False, top=False, labelbottom=False, labeltop=False)  # Turn off ticks and tick labels for ax2
ax.set_xlabel(r'$t$', fontsize=20)
# Customize the primary axis tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
ax.tick_params(axis='both', which='minor', labelsize=16)  # Change minor tick label size
fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

# Create a single legend
plt.savefig(os.path.join(base_args.fig_dir, f'measurement_prediction_{base_args.index}.png'))
plt.close()
