import numpy as np
import uuid
import torch
import argparse
import os
import random
import time
import json
from trendy.data._pdes import *
from trendy.data._utils import *
from trendy.models import TRENDy
from trendy.train import set_seed
from itertools import product
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True)
parser.add_argument('--num_samples', type=int, required=True, help='Total number of samples divided between train, val and test')
parser.add_argument('--chunks', type=int, required=True, help='Total number of SLURM tasks')
parser.add_argument('--pde_class', default='GrayScott', type=str, help='PDE family')
parser.add_argument('--nx', type=int, default=64, help='Spatial resolution of data.')
parser.add_argument('--num_channels', type=int, default=2, help='Number of channels in data.')
parser.add_argument('--num_inits', type=int, default=1, help='Number of intitial conditions per sample.')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--measurement_type', type=str, default='channel_average', help='Which measurement to use.')
parser.add_argument('--scattering_J', type=int, default=2, help='Scattering scale parameter.')
parser.add_argument('--scattering_L', type=int, default=8, help='Scattering orientation parameter.')
parser.add_argument('--scattering_max_order', type=int, default=2, help='Scattering max order parameter.')
parser.add_argument('--save_pde_solutions', action='store_true', help='If enabled, saves the raw PDE solution images as well.')
parser.add_argument('--noise_type', type=str, default=None)
parser.add_argument('--randomized_dims', nargs='+', type=int, default=[0,1])
parser.add_argument('--val_size', type=float, default=.2)
parser.add_argument('--epsilon', type=float, default=0.0)
args = parser.parse_args()

# Make output dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# Save training arguments for reproducibility
args_dict = vars(args)
with open(os.path.join(args.output_dir, 'data_manifest.json'), 'w') as json_file:
    json.dump(args_dict, json_file, indent=4)

set_seed(args.task_id)

# Ensure output directory exists
train_output_dir = os.path.join(args.output_dir, 'train')
val_output_dir = os.path.join(args.output_dir, 'val')
test_output_dir = os.path.join(args.output_dir, 'test')
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)
all_output_dirs = {'train': train_output_dir, 'validation': val_output_dir, 'test': test_output_dir}

# Function to save data
def save_data(solution, mode, params, label, solver_params, raw_solution=None):
    # Get output dir
    output_dir = all_output_dirs[mode]

    # Get id
    unique_id = str(uuid.uuid4())

    # Save
    torch.save(solution, os.path.join(output_dir, f'X_{unique_id}.pt'))
    torch.save(params, os.path.join(output_dir, f'p_{unique_id}.pt'))
    torch.save(label, os.path.join(output_dir, f'y_{unique_id}.pt'))
    if raw_solution is not None:
        torch.save(raw_solution[0], os.path.join(output_dir, f'I_{unique_id}.pt'))
        torch.save(raw_solution[-1], os.path.join(output_dir, f'U_{unique_id}.pt'))

# Generate and save data for this task's portion
start = time.time()
generated = 0

# Get parameters
with open('../trendy/data/pde_configurations.json', 'r') as f:
    pde_configurations = json.load(f)

config = pde_configurations[args.pde_class]
param_ranges = config["recommended_param_ranges"]
bifurcation_func = eval(f"lambda x : {config['bifurcation_func']}")

# Get param generator
param_generator = ParameterGenerator(param_ranges, randomized_dims=args.randomized_dims, val_size=args.val_size, epsilon=args.epsilon, curve_func=bifurcation_func)

# Noise
if args.noise_type is not None:
    occluder = Occluder(args.noise_type)

# Calculate start and end indices for each chunk
samples_per_chunk = args.num_samples // args.chunks

in_shape = [args.num_channels, args.nx, args.nx]
measurement_kwargs = {'J': args.scattering_J, 'L':args.scattering_L, 'max_order': args.scattering_max_order}

model = TRENDy(in_shape, measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_pca=False)

total_mem = 0
for i in range(samples_per_chunk):

    # Randomly select a PDE class name and create an instance
    params, mode = param_generator.generate_sample()
    solution = None
    while solution is None:
        pde = PDE(args.pde_class, params=torch.tensor(params))
        solution = pde.run(num_inits=1, verbose=1, train=False, new_solver=True).squeeze()

        if solution is None:
            print(f'Found a bad one: {params}.', flush=True)

    if args.noise_type is not None:
        solution = occluder.apply_occlusion(solution)

    if args.save_pde_solutions:
        raw_init = solution[0].clone() if args.save_pde_solutions else None
        raw_final = solution[-1].clone() if args.save_pde_solutions else None

    solution = model.compute_measurement(solution.unsqueeze(0)).clone().squeeze()
    print(i, flush=True)

    generated += 1

    # Save data
    save_data(solution, mode, pde.flat_params, args.pde_class, pde.config['solver_params'], raw_solution=[raw_init, raw_final])

stop = time.time()
print(f"Generated {generated} samples in {stop - start} seconds for Task ID: {args.task_id}.")
