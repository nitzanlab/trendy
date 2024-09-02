import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.utils import save_solution_as_movie
from time import time
import numpy as np
import os
import warnings
import argparse

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--pde_name', type=str, default='GrayScott')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--params', nargs='+', type=float, default=None, help='System to parameters, if not random.')
parser.add_argument('--fig_dir', type=str, default='./figs')
parser.add_argument('--step', type=int, default=1, help='Plotting period for video.')
args = parser.parse_args()

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
print(f'Using {num_cpus} cpus')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data preparation
if args.data_dir is None:
    if args.params is None:
        pde = PDE(args.pde_name)
    else:
        pde = PDE(args.pde_name, params=torch.tensor(args.params))
    print(f"Solving {args.pde_name} PDE.", flush=True)
    pde_solution = pde.run().squeeze()
else:
    fn = os.path.join(args.data_dir, f'X_{args.index}.pt')
    pde_solution = torch.load(fn)

file_name = f'{args.pde_name}_movie.gif' if args.data_dir is None else f'{args.pde_name}_movie_{args.index}.gif'
print("Saving movie.", flush=True)
save_solution_as_movie(pde_solution, args.fig_dir, file_name=file_name, step=args.step)
