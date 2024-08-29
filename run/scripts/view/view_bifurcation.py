import torch
from torch.utils.data import DataLoader
from phase2vec_spatial import PDE
from phase2vec_spatial.models import TRENDy
from phase2vec_spatial.train import *
from phase2vec_spatial.data import SP2VDataset
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
parser.add_argument('--base_params', nargs='+', required=True, type=float)
parser.add_argument('--num_sols', type=int, default=100)
parser.add_argument('--bp_ind', type=int, required=True)
parser.add_argument('--bp_min', type=float, required=True)
parser.add_argument('--bp_max', type=float, required=True)
parser.add_argument('--fig_dir', type=str, default='./figs', help='Where to save figures')
base_args = parser.parse_args()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

manifest_fn = os.path.join(base_args.model_dir, 'training_manifest.json')
with open(manifest_fn, 'r') as f:
    training_data = json.load(f)

args = argparse.Namespace()
for key, value in training_data.items():
    setattr(args, key, value)

measurement_kwargs = {'device':device}
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_pca=args.use_pca, pca_components=args.pca_components, num_params=len(base_args.base_params), node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous)
model, _, _ = load_checkpoint(model, base_args.model_dir, device=device)

model = model.to(device)

par_range = torch.linspace(base_args.bp_min, base_args.bp_max, base_args.num_sols)
all_gt = []
all_est = []
all_params = []

print_every = 5
print(f'Solving {base_args.num_sols} instances of {base_args.pde_name} while varying parameter {base_args.bp_ind}.', flush=True)
for s in range(base_args.num_sols):

    if s % print_every == 0:
        print(f'Solution {s}. Parameter: {par_range[s]:.3f}.', flush=True)

    # Set current parameter
    params = torch.tensor(base_args.base_params)
    params[base_args.bp_ind] = par_range[s]
    pde = PDE(base_args.pde_name, params=params)

    # Compute gt measurement
    solution = model.compute_measurement(pde.run().to(device))

    # Compute est measurement
    init = solution[0]
    estimated = model.run(init, params.unsqueeze(0))

    # Collate
    all_gt.append(solution)
    all_est.append(estimated)
    all_params.append(params)
print('Done', flush=True)
print('Making videos for...', flush=True)
for plot_sols, plot_which in zip([all_gt, all_est], ['gt', 'est']):
    print(plot_which)
    num_samples, time_steps, num_features = plot_sols.shape
    
    # Set up the figure, the axis, and the plot elements
    fig, ax = plt.subplots()
    
    linestyle ='-' if plot_which == 'gt' else ':'
    lines2 = [ax.plot(np.arange(time_steps), plot_sols[0,:,i], linestyle=linestyle)[0] for i in range(num_features)]
    
    # Set up the axes limits
    ax.set_xlim(0, time_steps)
    ax.set_ylim(np.min([plot_sols]), np.max([plot_sols]))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$S_i$')
    
    def init():
        for line in lines2:
            line.set_data([],[])
        return lines2
    
    def update(frame):
        for i, line in enumerate(lines2):
            current = plot_sols[frame, :, i]
            line.set_data(np.arange(time_steps), current)
        param = all_params[frame][1].item()
        ax.set_title(fr'$B={param}$')
        return lines2
    
    ani = FuncAnimation(fig, update, frames=num_samples, init_func=init, blit=True)
    
    # saving to m4 using ffmpeg writer 
    writervideo = FFMpegWriter(fps=10) 
    print('Saving video...', flush=True)
    ani.save(os.path.join(base_args.fig_dir, f'{plot_which}_{args.pde_name}.mp4'), writer=writervideo) 
    plt.close()
