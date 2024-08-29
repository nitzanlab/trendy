import numpy as np
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
from torchvision import transforms
from itertools import product
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Where min data is stored.')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--measurement_type', type=str, default='channel_average', help='Which measurement to use.')
parser.add_argument('--scattering_J', type=int, default=2, help='Scattering scale parameter.')
parser.add_argument('--scattering_L', type=int, default=8, help='Scattering orientation parameter.')
parser.add_argument('--scattering_max_order', type=int, default=2, help='Scattering max order parameter.')
parser.add_argument('--save_pde_solutions', action='store_true', help='If enabled, saves the raw PDE solution images as well.')
parser.add_argument('--task_id', type=int, required=True)
parser.add_argument('--chunks', type=int, required=True)
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

# Ensure output directory exists
train_output_dir = os.path.join(args.output_dir, 'train')
test_output_dir = os.path.join(args.output_dir, 'test')
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Function to save data
def save_data(index, solution, params, label, mode, raw_solution=None):
    output_dir = train_output_dir if mode == 'train' else test_output_dir
    torch.save(solution, os.path.join(output_dir, f'X_{index}.pt'))
    torch.save(params, os.path.join(output_dir, f'p_{index}.pt'))
    torch.save(label, os.path.join(output_dir, f'y_{index}.pt'))
    if raw_solution is not None:
        torch.save(raw_solution, os.path.join(output_dir, f'U_{index}.pt'))

# Generate and save data for this task's portion
print(f"Task ID: {args.task_id}")
generated = 0

measurement_kwargs = {'J': args.scattering_J, 'L':args.scattering_L, 'max_order': args.scattering_max_order}

# Dummy shape to start
in_shape = [2, 64, 64]
model = TRENDy(in_shape, measurement_type=args.measurement_type, measurement_kwargs=measurement_kwargs, use_pca=False)

# Get all data
train_fns = glob.glob(os.path.join(args.data_dir, 'train', 'X_*'))
test_fns = glob.glob(os.path.join(args.data_dir, 'test', 'X_*'))
all_fns = train_fns + test_fns

fns_per_chunk = int(len(all_fns) // args.chunks)
task_inds     = np.arange(fns_per_chunk * (args.task_id-1), min(fns_per_chunk * args.task_id, len(all_fns) + 1))

task_fns = [all_fns[ind] for ind in task_inds]

start = time.time()
for ind, fn in zip(task_inds,task_fns):

    mode = 'train' if 'train' in fn else 'test'

    # Load data
    index = int(re.search(r'X_(\d+)\.pt', fn).group(1))
    solution = torch.load(fn).float()
    p_dict = torch.load(os.path.join(args.data_dir, mode, f'p_{index}.pt'))
    p = torch.tensor([p_dict['e_levels'], p_dict['d_levels']]).float()

    # Set measurement info for this data (changes with shape)
    in_shape = list(solution.shape)
    model.set_measurement(args.measurement_type, in_shape, measurement_kwargs)

    if args.save_pde_solutions:
        raw_solution = solution[-1].clone() if args.save_pde_solutions else None
    
    solution = model.compute_measurement(solution.unsqueeze(0)).clone().squeeze()
    print(ind, flush=True)
    generated += 1
    
    # Save data
    save_data(index, solution, p, 'min', mode, raw_solution=raw_solution)
    
stop = time.time()
print(f"Generated {generated} samples in {stop - start} seconds for Task ID: {args.task_id}.")
