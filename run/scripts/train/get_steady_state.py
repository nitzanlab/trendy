import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import TRENDy
from trendy.train import *
from trendy.data import SP2VDataset
from torchvision import models
from scipy.ndimage import correlate
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from time import time
import numpy as np
plt.style.use('ggplot')
import os
import warnings
import argparse
from joblib import load
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--pde_name', type=str, default='Brusselator')
parser.add_argument('--model_dir', type=str, default='./models/tentative/run_0')
parser.add_argument('--num_params', type=int, default=4)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./data', help='Where to save figures')
base_args = parser.parse_args()

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

manifest_fn = os.path.join(base_args.model_dir, 'training_manifest.json')
with open(manifest_fn, 'r') as f:
    training_data = json.load(f)

args = argparse.Namespace()
for key, value in training_data.items():
    setattr(args, key, value)

measurement_kwargs = {'device':device}
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_pca=args.use_pca, pca_components=args.pca_components, num_params=base_args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous, pca_dir=args.pca_dir)
model, _, _ = load_checkpoint(model, base_args.model_dir, device=device, pca_dir=args.pca_dir)

model = model.to(device)

all_gt = []
all_est = []
all_params = []
all_ss = []

print(f'Acquiring {base_args.num_samples} steady states from {base_args.pde_name}', flush=True)

count = 0
while count < base_args.num_samples:

    print(f'Current count: {count}', flush=True)

    start = time()

    # Set current parameter
    pde = PDE(base_args.pde_name, device=device)
    params = pde.flat_params

    bifurcation_func = lambda x : eval(pde.config['bifurcation_func'])

    if params[1] > bifurcation_func(params):
        print('Found a post-bifurcation example. Skipping!', flush=True)
        continue
    else:
        print('Got a good example.', flush=True)

    # Compute gt measurement
    pde_solution = pde.run().squeeze()
    solution = model.compute_measurement(pde_solution.unsqueeze(0))

    # Collate
    all_ss.append(solution.squeeze()[-1].detach().cpu().numpy())
    all_params.append(params.detach().cpu().numpy())

    count += 1
    
print('Done', flush=True)

print(f'Saving steady states in {base_args.save_dir}', flush=True)
all_ss = np.array(all_ss)
all_params = np.array(all_params)

np.save(os.path.join(base_args.save_dir, 'all_ss.npy'), all_ss)
np.save(os.path.join(base_args.save_dir, 'all_params.npy'), all_params)
