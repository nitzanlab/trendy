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
# Parameters
# Parameters
x_values = np.linspace(1.5, 4.0, 100)
n_samples = 50  # Number of samples per batch
mu = 0.1  # Magnitude of perturbation

# Prepare the input sequence
jacobians = []
print("Input sequence prepared", flush=True)

for x in x_values:
    vector1 = torch.tensor([[1.5, x / 1.5]], dtype=torch.float32)
    vector2 = torch.tensor([[1.5, x, 0.1, 0.2]], dtype=torch.float32)

    # Generate batch of perturbed inputs
    eps = mu * (2 * torch.rand(n_samples, 2) - 1)  # Random perturbations
    perturbed_vector1_batch = vector1 + eps  # Broadcasting addition
    perturbed_vector1_batch.requires_grad = True  # Ensure this is a leaf tensor

    # Expand vector2 and concatenate
    vector2_expanded = vector2.expand(n_samples, -1)
    input_concat_batch = torch.cat((perturbed_vector1_batch, vector2_expanded), dim=1)

    # Forward pass
    output_batch = model(input_concat_batch)[:, :2]  # Select only the first two dimensions

    # Compute the Jacobian with respect to perturbed_vector1_batch
    jacobians_batch = []
    for i in range(output_batch.shape[1]):
        grad_output = torch.zeros_like(output_batch)
        grad_output[:, i] = 1
        output_batch.backward(grad_output, retain_graph=True)
        jacobians_batch.append(perturbed_vector1_batch.grad.detach().clone())
        perturbed_vector1_batch.grad.zero_()

    jacobians_batch = torch.stack(jacobians_batch, dim=1).squeeze()
    jacobians.append(jacobians_batch)

jacobians = torch.stack(jacobians, dim=0)
print("Jacobians computed", flush=True)

# Compute the eigenvalues of each Jacobian
eigenvalues = []

for batch in jacobians:
    batch_eigenvalues = []
    for jacobian in batch:
        jacobian_np = jacobian.numpy()
        batch_eigenvalues.append(np.linalg.eigvals(jacobian_np))
    eigenvalues.append(np.array(batch_eigenvalues))

eigenvalues = np.array(eigenvalues, dtype=complex)
print("Eigenvalues computed", flush=True)

# Separate real and complex eigenvalues
real_eigenvalues = []
complex_eigenvalues = []

for batch in eigenvalues:
    real_eigenvalues.append([eig for eig in batch if np.isreal(eig).all()])
    complex_eigenvalues.append([eig for eig in batch if not np.isreal(eig).all()])

real_eigenvalues = np.array(real_eigenvalues, dtype=float)
complex_eigenvalues = np.array(complex_eigenvalues, dtype=complex)

# Compute mean and standard error of eigenvalues
mean_real_parts = np.mean(real_eigenvalues, axis=1)
std_error_real_parts = np.std(real_eigenvalues, axis=1) / np.sqrt(n_samples)
mean_complex_real_parts = np.mean(complex_eigenvalues.real, axis=1)
std_error_complex_real_parts = np.std(complex_eigenvalues.real, axis=1) / np.sqrt(n_samples)
mean_complex_imag_parts = np.mean(complex_eigenvalues.imag, axis=1)
std_error_complex_imag_parts = np.std(complex_eigenvalues.imag, axis=1) / np.sqrt(n_samples)

# Check for complex conjugate symmetry in complex eigenvalues
for i in range(len(x_values)):
    if not np.allclose(mean_complex_real_parts[i, 0], mean_complex_real_parts[i, 1]):
        print(f"Complex real parts not equal at x = {x_values[i]}: {mean_complex_real_parts[i, 0]}, {mean_complex_real_parts[i, 1]}", flush=True)
    if not np.allclose(mean_complex_imag_parts[i, 0], -mean_complex_imag_parts[i, 1]):
        print(f"Imaginary parts not equal and opposite at x = {x_values[i]}: {mean_complex_imag_parts[i, 0]}, {mean_complex_imag_parts[i, 1]}", flush=True)

# Plot the real and imaginary parts of the eigenvalues
plt.figure(figsize=(10, 6))

# Plot real parts of real eigenvalues
plt.plot(x_values, mean_real_parts[:, 0], label='Mean Real Part of Real Eigenvalue 1', color='green', linestyle='-')
plt.fill_between(x_values, mean_real_parts[:, 0] - std_error_real_parts[:, 0],
                 mean_real_parts[:, 0] + std_error_real_parts[:, 0], color='green', alpha=0.2)
plt.plot(x_values, mean_real_parts[:, 1], label='Mean Real Part of Real Eigenvalue 2', color='purple', linestyle='-')
plt.fill_between(x_values, mean_real_parts[:, 1] - std_error_real_parts[:, 1],
                 mean_real_parts[:, 1] + std_error_real_parts[:, 1], color='purple', alpha=0.2)

# Plot real parts of complex eigenvalues
plt.plot(x_values, mean_complex_real_parts[:, 0], label='Mean Real Part of Complex Eigenvalue 1', color='red', linestyle='-')
plt.fill_between(x_values, mean_complex_real_parts[:, 0] - std_error_complex_real_parts[:, 0],
                 mean_complex_real_parts[:, 0] + std_error_complex_real_parts[:, 0], color='red', alpha=0.2)
plt.plot(x_values, mean_complex_real_parts[:, 1], label='Mean Real Part of Complex Eigenvalue 2', color='blue', linestyle='-')
plt.fill_between(x_values, mean_complex_real_parts[:, 1] - std_error_complex_real_parts[:, 1],
                 mean_complex_real_parts[:, 1] + std_error_complex_real_parts[:, 1], color='blue', alpha=0.2)

# Plot imaginary parts of complex eigenvalues
plt.plot(x_values, mean_complex_imag_parts[:, 0], label='Mean Imag Part of Complex Eigenvalue 1', color='red', linestyle='--')
plt.fill_between(x_values, mean_complex_imag_parts[:, 0] - std_error_complex_imag_parts[:, 0],
                 mean_complex_imag_parts[:, 0] + std_error_complex_imag_parts[:, 0], color='red', alpha=0.2)
plt.plot(x_values, mean_complex_imag_parts[:, 1], label='Mean Imag Part of Complex Eigenvalue 2', color='blue', linestyle='--')
plt.fill_between(x_values, mean_complex_imag_parts[:, 1] - std_error_complex_imag_parts[:, 1],
                 mean_complex_imag_parts[:, 1] + std_error_complex_imag_parts[:, 1], color='blue', alpha=0.2)

plt.xlabel('x')
plt.ylabel('Eigenvalues')
plt.title('Mean Real and Imaginary Parts of the Eigenvalues of the Jacobians with Standard Error')
plt.legend()
plt.grid(True)

# Ensure the directory exists
os.makedirs('./figs/', exist_ok=True)

# Save the plot
plt.savefig('./figs/eigenvalues_real_and_imag_parts.png')
print("Plot saved to ./figs/eigenvalues_real_and_imag_parts.png", flush=True)
