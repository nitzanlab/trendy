import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import TRENDy
from trendy.train import *
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
parser.add_argument('--model_dir', type=str, default='./models/tentative/run_0')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--clip_target', type=int, default=-1)
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
print(f'Using {num_cpus} cpus', flush=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}', flush=True)

# Data preparation
target = torch.load(os.path.join(args.data_dir, base_args.mode,  f'X_{base_args.index}.pt')).float().unsqueeze(0).to(device)
params = torch.load(os.path.join(args.data_dir, base_args.mode, f'p_{base_args.index}.pt')).float().unsqueeze(0).to(device)

target = target[:,:args.clip_target]

# Load model
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_log_scale=args.log_scale ,use_pca=args.use_pca, pca_components=args.pca_components, num_params=params.shape[-1], node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous, pca_dir=args.pca_dir).to(device)
model, _, _ = load_checkpoint(model, base_args.model_dir, device=device, pca_dir=args.pca_dir)

print(f'Loaded model of architecture: {model}', flush=True)

# Seed model and run
init   = target[:,0]
est = model.run(init, params).detach()[:,:args.clip_target]

# Compute pca of full solution, if necessary
if args.log_scale:
    target = torch.log10(target)
if args.use_pca:
    b, t, f  = target.shape
    target = target.reshape(-1,f)
    target = model.pca_layer(target).reshape(b,t,-1)
target = target.detach()

# Loss
criterion = NthOrderLoss(dt_est = args.dt_est, dt_true = args.dt_true, der_weight=args.der_weight, burn_in_size=args.burn_in_size)
loss = criterion(est, target).mean()
print(loss.item())

est = est.detach().cpu().numpy()
target = target.detach().cpu().numpy()

fig, ax = plt.subplots()
num_colors = model.node_input_dim
all_colors  = hsv_to_rgb(np.column_stack([np.linspace(0, 1, num_colors, endpoint=False), np.ones((num_colors, 2))]))
gt_lines = []
for i in range(num_colors):
    line, = ax.plot(target[0,:,i], label = fr'GT $\tilde{{S}}_{{{i}}}$', color=all_colors[i])
    gt_lines.append(line)

#ax.set_title(f'Loss : {loss:.4f}')

ax2 = ax.twiny()
est_lines = []
for i in range(num_colors):
    line, = ax2.plot(est[0,:,i], label = fr'Trendy $\tilde{{S}}_{{{i}}}$', linestyle='--', color=all_colors[i])
    est_lines.append(line)
ax2.grid(False)  # Turn off the grid for ax2
ax2.tick_params(which='both', bottom=False, top=False, labelbottom=False, labeltop=False)  # Turn off ticks and tick labels for ax2
ax.set_xlabel(r'$t$', fontsize=20)
# Customize the primary axis tick label sizes
ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
ax.tick_params(axis='both', which='minor', labelsize=16)  # Change minor tick label size
fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

# Create a single legend
fn = f'measurement_prediction_{base_args.index}.png'
plt.savefig(os.path.join(base_args.fig_dir, fn))
plt.close()
print(f'Prediction saved at {fn}.', flush=True)
