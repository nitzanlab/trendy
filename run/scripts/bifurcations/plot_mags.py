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
from scipy.stats import binned_statistic
from time import time
import numpy as np
plt.style.use('ggplot')
import os
import warnings
import argparse
from joblib import load
import matplotlib.colors as mcolors

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='Which model to use, if any. If None, then use true measurement.')
parser.add_argument('--model_type', default='node', type=str, help='Which type of model to use.')
parser.add_argument('--measurement_type', default='all_channel_average', type=str, help='Which measurement to use.')
parser.add_argument('--input_dim', default=2, type=int, help='Measurement spapce dim.')
parser.add_argument('--aug_dim', default=4, type=int, help='Parameter spapce dim.')
parser.add_argument('--num_layers', default=4, type=int, help='Number of NODE layers.')
parser.add_argument('--num_features', default=64, type=int, help='Number of NODE features.')
parser.add_argument('--fig_dir', type=str, required=True, help='Where to save figures')
args = parser.parse_args()

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
#num_cpus = os.cpu_count()
print(f'Using {num_cpus} cpus')
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

full_model = get_model(args.model_type, args.input_dim, args.aug_dim, args.num_layers, args.num_features).to(device)
full_model = load_checkpoint(full_model, args.model_path)

model = full_model.ode_func
# Define the input space
A = 1.5
B_values = np.linspace(1, 4, 100)
xy_values = np.linspace(1, 4, 100)
Du = 0.1
Dv = 0.2

# Function to compute the magnitude of the ODE output in batches
def compute_ode_magnitude_batch(model, x_values, y_values, A, B_values, Du, Dv, batch_size=1000):
    magnitudes = []
    argmin_x = []
    argmin_y = []
    
    for B in B_values:
        batch_points = np.array([[x, y, A, B, Du, Dv] for x in x_values for y in y_values])
        input_tensors = torch.tensor(batch_points, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(input_tensors)
        batch_magnitudes = torch.norm(outputs[:, :2], dim=1).numpy()
        magnitudes.append(np.min(batch_magnitudes))
        
        min_index = np.argmin(batch_magnitudes)
        min_point = batch_points[min_index]
        argmin_x.append(min_point[0])
        argmin_y.append(min_point[1])
    
    return np.array(magnitudes), np.array(argmin_x), np.array(argmin_y)

# Compute minimum magnitudes and argmin x, y for each B
min_magnitudes, argmin_x, argmin_y = compute_ode_magnitude_batch(
    model, xy_values, xy_values, A, B_values, Du, Dv)

# Compute magnitudes along the true equilibria manifold
true_points = np.array([[1.5, B/1.5, 1.5, B, 0.1, 0.2] for B in B_values])
true_input_tensors = torch.tensor(true_points, dtype=torch.float32)

with torch.no_grad():
    true_outputs = model(true_input_tensors)
true_magnitudes = torch.norm(true_outputs[:, :2], dim=1).numpy()

# True equilibria points
true_B = B_values
true_x = np.full_like(B_values, 1.5)
true_y = B_values / 1.5

# Create directory for figures
os.makedirs('./figs', exist_ok=True)

# Plot the 3D curve with swapped axes and true equilibria
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(B_values, argmin_y, argmin_x, 'b-', label='Equilibria Curve')
ax.plot(true_B, true_y, true_x, 'r--', label='True Equilibria')
ax.set_xlabel('B')
ax.set_ylabel('y')
ax.set_zlabel('x')
ax.set_title('Equilibria as B varies')
plt.legend()
plt.savefig('./figs/equilibria_3d_plot.png')
plt.show()

# Plot the two-panel figure with B vs x and B vs y
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# B vs x plot
ax1.plot(B_values, argmin_x, 'b-', label='Equilibria Curve')
ax1.plot(true_B, true_x, 'r--', label='True Equilibria')
ax1.set_xlabel('B')
ax1.set_ylabel('x')
ax1.set_title('B vs x')
ax1.legend()

# B vs y plot
ax2.plot(B_values, argmin_y, 'b-', label='Equilibria Curve')
ax2.plot(true_B, true_y, 'r--', label='True Equilibria')
ax2.set_xlabel('B')
ax2.set_ylabel('y')
ax2.set_title('B vs y')
ax2.legend()

plt.tight_layout()
plt.savefig('./figs/equilibria_b_vs_x_y.png')
plt.show()

# Plot magnitudes along the true and actual equilibria
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(B_values, min_magnitudes, 'b-', label='Actual Equilibria Magnitudes')
ax.plot(B_values, true_magnitudes, 'r--', label='True Equilibria Magnitudes')
ax.set_xlabel('B')
ax.set_ylabel('Magnitude')
ax.set_title('Magnitudes along Equilibria')
ax.legend()
plt.savefig('./figs/equilibria_magnitudes.png')
plt.show()
