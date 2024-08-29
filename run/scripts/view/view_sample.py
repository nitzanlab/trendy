import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import TRENDy
from trendy.data import SP2VDataset
from trendy.train import *
import matplotlib.pyplot as plt
from time import time
import numpy as np
#plt.style.use('ggplot')
import os
import warnings
import argparse

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--pde_name', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--model_dir', type=str, default='./models/nodes')
parser.add_argument('--measurement_type', type=str, default='scattering')
parser.add_argument('--in_shape', nargs='+', type=int, default = [2, 64, 64])
parser.add_argument('--num_params', type=int, default=4)
parser.add_argument('--node_hidden_layers', nargs='+', type=int, default=[64,64,64,64])
parser.add_argument('--node_activations', nargs='+', default='relu')
parser.add_argument('--use_pca', action="store_true")
parser.add_argument('--pca_components', type=int, default=2)
parser.add_argument('--dt_est', type=float,default=1e-2)
parser.add_argument('--T_est', type=float,default=1.0)
parser.add_argument('--log_scale', action="store_true")
parser.add_argument('--fig_dir', type=str, default='./figs')
args = parser.parse_args()

if args.pde_name is None:
    setattr(args, 'pde_name', args.data_dir.split('/')[-2])

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
print(f'Using {num_cpus} cpus')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load model
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_log_scale=args.log_scale, use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est).to(device)
if args.use_pca:
    model, _, _ = load_checkpoint(model, args.model_dir, device=device)

# Data preparation
if args.data_dir is None:
    pde = PDE(args.pde_name)
    params = pde.flat_params
    print(f"Solving {args.pde_name} PDE.")
    pde_solution = pde.run()
    final_pde_state = pde_solution[-1].detach().numpy()
    gt_solution = model.compute_measurement(pde_solution).squeeze().detach().numpy()
else:
    print(f"Loading data from {args.data_dir}.")
    # Load solution
    gt_solution = torch.load(os.path.join(args.data_dir, f'X_{args.index}.pt')).float().squeeze().to(device)
    print(f'Loaded data of shape {gt_solution.shape}', flush=True)

    # Load params
    params = torch.load(os.path.join(args.data_dir, f'p_{args.index}.pt')).to(device)

    # If loaded solution is a video
    if len(gt_solution.shape) == 4:
        final_pde_state = gt_solution[-1].unsqueeze(0)
        gt_solution = model.compute_measurement(gt_solution.unsqueeze(0)).squeeze().detach().numpy()
    # Otherwise, if it's a measurement
    else:
        # Compute pca of full solution, if necessary
        if args.log_scale:
            gt_solution = torch.log10(gt_solution)
        if args.use_pca:
            gt_solution = model.pca_layer(gt_solution)
        gt_solution = gt_solution.cpu().detach().numpy()

        # Load final_pde_state, if it exists
        U_path = os.path.join(args.data_dir, f'U_{args.index}.pt')
        final_pde_state = torch.load(U_path) if os.path.exists(U_path) else None

#TODO: remove
np.save('./data/min_sample.npy', gt_solution)

print("Done\n")
if final_pde_state is None:
    fig, ax = plt.subplots()

    ax.plot(gt_solution)
    ax.set_xlabel(r'$t$')
else:
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(final_pde_state[0])
    axes[1].plot(gt_solution)
    axes[1].set_xlabel(r'$t$')
axes[0].set_title(params.cpu().numpy())

fn = f'{args.pde_name}_sample_{args.index}.png' if args.data_dir is not None else f'{args.pde_name}_sample.png'
plt.savefig(os.path.join(args.fig_dir, fn))
plt.close()
print(f'Saved figure at {fn}.', flush=True)
