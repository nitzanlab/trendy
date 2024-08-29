import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import get_model, load_checkpoint
from trendy.data import SP2VDataset
from trendy.data._measurements import *
from torchvision import models
from scipy.stats import mode
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
parser.add_argument('--pde_name', type=str, default='GrayScott')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--model_dir', type=str, default='./models/nodes')
parser.add_argument('--measurement_type', type=str, default='scattering')
parser.add_argument('--in_shape', nargs='+', default = [2, 64, 64])
parser.add_argument('--base_params', nargs='+', required=True)
parser.add_argument('--node_hidden_layers', nargs='+', default=[64,64,64,64])
parser.add_argument('--node_activations', type=str, default='relu')
parser.add_argument('--use_pca', action="store_true")
parser.add_argument('--pca_components', type=int, default=2)
parser.add_argument('--dt_est', type=float,default=1e-2)
parser.add_argument('--T_est', type=float,default=1.0)
parser.add_argument('--num_sols', type=int, default=100)
parser.add_argument('--bp_ind', type=int, required=True)
parser.add_argument('--bp_min', type=float, required=True)
parser.add_argument('--bp_max', type=float, required=True)
parser.add_argument('--fig_dir', type=str, required=True, help='Where to save figures')
args = parser.parse_args()

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
#num_cpus = os.cpu_count()
print(f'Using {num_cpus} cpus')
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_pca=args.use_pca, pca_components=args.pca_components, num_params=len(args.base_params), node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est)

par_range = torch.linspace(args.bp_min, args.bp_max, args.num_sols)
all_gt = []
all_est = []
all_params = []
for s in range(num_sols):

    # Set current parameter
    params = torch.tensor(args.base_params)
    params[args.bp_ind] = par_range[s]
    pde = PDE(args.pde_name, params=params)

    # Compute gt measurement
    solution = model.compute_measurement(pde.run())

    # Compute est measurement
    init = solution[0]
    estimated = model.run(init, params)

    # Collate
    all_gt.append(solution)
    all_est.append(estimated)
    all_params.append(params)

for plot_sols, plot_which in zip([all_gt, all_est], ['gt', 'est'])
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
    ani.save(os.path.join(args.fig_dir, f'{plot_which}_{args.pde_name}.mp4'), writer=writervideo) 
    plt.close()
