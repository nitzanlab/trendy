import numpy as np
import torch
import argparse
import os
import random
import time
import json
from trendy.data._pdes import *
from trendy.data._utils import *
from trendy.data._measurements import *
from trendy.models import TRENDy
from trendy.train import set_seed
from itertools import product
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, help='SLURM array task ID')
parser.add_argument('--chunks', type=int, required=True, help='Total number of SLURM tasks')
parser.add_argument('--num_bins', type=int, default=50, help='Total number of samples')
parser.add_argument('--pde_class', default='GrayScott', type=str, help='PDE family')
parser.add_argument('--nx', type=int, default=64, help='Spatial resolution of data.')
parser.add_argument('--num_channels', type=int, default=2, help='Number of channels in data.')
parser.add_argument('--num_inits', type=int, default=1, help='Number of intitial conditions per sample.')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test split')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--vary_diffusion', action='store_true', help='Enable varying diffusion process.')
parser.add_argument('--measurement_type', type=str, default='channel_average', help='Which measurement to use.')
parser.add_argument('--scattering_J', type=int, default=2, help='Scattering scale parameter.')
parser.add_argument('--scattering_L', type=int, default=8, help='Scattering orientation parameter.')
parser.add_argument('--scattering_max_order', type=int, default=2, help='Scattering max order parameter.')
parser.add_argument('--save_pde_solutions', action='store_true', help='If enabled, saves the raw PDE solution images as well.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Make output dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Save training arguments for reproducibility
args_dict = vars(args)
with open(os.path.join(args.output_dir, 'data_manifest.json'), 'w') as json_file:
    json.dump(args_dict, json_file, indent=4)

# Set a known seed to get cross-task values
set_seed(args.seed)
rand_inds = np.random.permutation(args.num_bins**2)

# Set a random seed based on the task ID for repeatability
np.random.seed(args.task_id)
torch.manual_seed(args.task_id)

# Ensure output directory exists
train_output_dir = os.path.join(args.output_dir, 'train')
test_output_dir = os.path.join(args.output_dir, 'test')
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Determine the number of test samples
num_samples = args.num_bins**2
num_test_samples = int(num_samples * args.test_size)
num_train_samples = num_samples - num_test_samples

# Function to save data
def save_data(index, solution, params, label, solver_params, data_type, raw_solution=None):
    output_dir = train_output_dir if data_type == 'train' else test_output_dir
    torch.save(solution, os.path.join(output_dir, f'X_{index}.pt'))
    torch.save(params, os.path.join(output_dir, f'p_{index}.pt'))
    torch.save(label, os.path.join(output_dir, f'y_{index}.pt'))
    if raw_solution is not None:
        torch.save(raw_solution, os.path.join(output_dir, f'U_{index}.pt'))

# Generate and save data for this task's portion
print(f"Task ID: {args.task_id}")
start = time.time()
generated = 0

# Adjust for task_id starting from 1 in SLURM
task_index = args.task_id - 1  # Convert to 0-based index

# Get parameters
with open('../trendy/data/pde_configurations.json', 'r') as f:
    pde_configurations = json.load(f)

config = pde_configurations[args.pde_class]
param_ranges = config["recommended_param_ranges"]

#reaction parameters
discretized_ranges = [np.linspace(start, end, args.num_bins) for start, end in param_ranges[:-2]]
#diffusion parameters
if not args.vary_diffusion:
    discretized_ranges += [[param_ranges[-2][-1]]] # max value of first diffusion
    discretized_ranges += [[param_ranges[-1][-1]]] # max value of second diffusion
    num_params = args.num_bins ** len(param_ranges[:-2])
else:
    discretized_ranges = [np.linspace(start, end, args.num_bins) for start, end in param_ranges[-2:]]
    num_params = args.num_bins ** len(param_ranges)

all_params = list(product(*discretized_ranges))
all_params = [all_params[i] for i in rand_inds]

# Calculate start and end indices for each chunk
samples_per_chunk = num_samples // args.chunks
start_idx = task_index * samples_per_chunk
end_idx = min(start_idx + samples_per_chunk, num_samples)  # Ensure not to exceed num_samples

in_shape = [args.num_channels, args.nx, args.nx]
measurement_kwargs = {'J': args.scattering_J, 'L':args.scattering_L, 'max_order': args.scattering_max_order}

model = TRENDy(in_shape, measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_pca=False)

total_mem = 0
for i in range(start_idx, end_idx):
    # Determine data type (train or test) and adjust save_index
    data_type = 'train' if i < num_train_samples else 'test'
    if data_type == 'train':
        save_index = i
    else:
        save_index = i - num_train_samples

    # Randomly select a PDE class name and create an instance
    params = all_params[i]
    solution = None
    while solution is None:
        pde = PDE(args.pde_class, params=torch.tensor(params))
        solution = pde.run(num_inits=1, verbose=1, train=False, new_solver=True).squeeze()
        if solution is None:
            print(f'Found a bad one: {params}.', flush=True)

    if args.save_pde_solutions:
        raw_solution = solution[-1].clone() if args.save_pde_solutions else None

    solution = model.compute_measurement(solution.unsqueeze(0)).clone().squeeze()
    print(i, flush=True)

    generated += 1

    # Save data
    save_data(save_index, solution, pde.flat_params, args.pde_class, pde.config['solver_params'], data_type, raw_solution=raw_solution)

stop = time.time()
print(f"Generated {generated} samples in {stop - start} seconds for Task ID: {args.task_id}.")
