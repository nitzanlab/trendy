import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.utils import save_solution_as_movie
from trendy.data import SP2VDataset
import matplotlib.pyplot as plt
from time import time
import numpy as np
plt.style.use('ggplot')
import os
import warnings
import argparse

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/min/train')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--fig_dir', type=str, default='./figs')
parser.add_argument('--step', type=int, default=1, help='Plotting period for video.')
args = parser.parse_args()

# Data preparation
min_path = os.path.join(os.path.join(args.data_dir, f'X_{index}.pt'))
pde_solution = torch.load(min_path)

save_solution_as_movie(pde_solution, args.fig_dir, file_name=f'{args.pde_name}_movie.gif', step=args.step)
