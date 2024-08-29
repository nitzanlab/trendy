import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from trendy import PDE
from trendy.utils import save_solution_as_movie
from trendy.data import SP2VDataset, UnitNormalize, SpatialResize
from trendy.models import TRENDy
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
parser.add_argument('--measurement_type', type=str, default='scattering')
parser.add_argument('--pca_components', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--resize_shape', type=int, default=64)
parser.add_argument('--log_scale', action='store_true')
parser.add_argument('--fig_dir', type=str, default='./figs')
args = parser.parse_args()

# Num cpus
num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
print(f'Using {num_cpus} cpus.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
transform = transforms.Compose([UnitNormalize(), SpatialResize((args.resize_shape, args.resize_shape))])
ds = SP2VDataset(data_dir=args.data_dir, transforms=transform)
dl = DataLoader(ds, batch_size=args.batch_size, num_workers=num_cpus, shuffle=False, drop_last=False)

# Measurement preparation
use_pca = args.pca_components is not None
measurement_kwargs = {'J':4, 'L':4,'device':device}
model = TRENDy([2,args.resize_shape, args.resize_shape], measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_log_scale=args.log_scale, use_pca=use_pca, pca_components=args.pca_components, num_params=2).to(device)

if use_pca:
    model.fit_pca(dl, max_samples=200)

min_path = os.path.join(os.path.join(args.data_dir, f'X_{args.index}.pt'))
pde_solution = torch.load(min_path).unsqueeze(0)
pde_solution = transform(pde_solution).to(device)

measurement = model.compute_measurement(pde_solution).squeeze().detach().numpy()

fig, ax = plt.subplots()
ax.set_xlabel(r'$t$', fontsize=16)
ax.set_ylabel(r'$S_i$', fontsize=16)
ax.plot(measurement)
fn = f'min_measurement_{args.index}.png'
plt.savefig(os.path.join(args.fig_dir, fn))
plt.close()

print(f'Min measurement saved at {fn}.')
