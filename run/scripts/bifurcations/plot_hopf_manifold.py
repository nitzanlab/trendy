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
A_min = 1.5
A_max = 4.0
B_min = A_min
B_max = 3 + A_max**2
A_values = np.linspace(A_min, A_max, 100)
B_values = np.linspace(B_min, B_max, 400)

hopf_manifold = []
for A in A_values:
    # Prepare the input sequence
    jacobians = []
    #print("Input sequence prepared", flush=True)
    
    for B in B_values:
        vector1 = torch.tensor([[A, B / A]], dtype=torch.float32, requires_grad=True)
        vector2 = torch.tensor([[A, B, 0.1, 0.2]], dtype=torch.float32)
    
        # Concatenate the vectors
        input_concat = torch.cat((vector1, vector2), dim=1)
    
        # Forward pass
        output = model(input_concat)[:, :2]  # Select only the first two dimensions
    
        # Compute the Jacobian with respect to vector1
        jacobian = []
        for i in range(output.shape[1]):
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian.append(vector1.grad.detach().clone())
            vector1.grad.zero_()
    
        jacobian = torch.stack(jacobian, dim=0).squeeze()
        jacobians.append(jacobian)
    
    jacobians = torch.stack(jacobians, dim=0)
    print("Jacobians computed", flush=True)
    
    # Compute the eigenvalues of each Jacobian
    eigenvalues = []
    
    for jacobian in jacobians:
        jacobian_np = jacobian.numpy()
        eigenvalues.append(np.linalg.eigvals(jacobian_np))
    
    eigenvalues = np.array(eigenvalues, dtype=complex)
    print("Eigenvalues computed", flush=True)
    
    # Extract real and imaginary parts
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    diff_real = np.abs(np.diff(real_parts, axis=0))
    B_ind = np.argmax(diff_real, axis=0)
    hopf_manifold.append(B_values[B_ind[1]])
    
    #hopf_candidates = []
    #for e, eigs in enumerate(eigenvalues):
    #    if np.imag(eigs[0]) != 0 and e > 0: #cplx eigenvalue
    #        if np.imag(eigenvalues[e-1][0]) > 0: # last one was too
    #            evs1 = eigenvalues[e-1]
    #            evs2 = eigs
    #
    #            r11 = np.real(evs1[0])
    #            r12 = np.real(evs1[1])
    #            r21 = np.real(evs2[0])
    #            r22 = np.real(evs2[1])
    #
    #            cond1 = np.sign(r11) != np.sign(r21) # first eval changes sign in real part
    #            cond2 = np.sign(r12) != np.sign(r22) # second eval changes sign in real part
    #
    #            if cond1 and cond2:
    #                hopf_candidates.append(B_values[e])
    #try:
    #    hopf_manifold.append(hopf_candidates[0])
    #except:
    #    hopf_manifold.append(np.nan)

A_train = [A for A in A_values if A < 2.5]
A_extra = [A for A in A_values if A >= 2.5]
hopf_manifold_train = hopf_manifold[:len(A_train)]
hopf_manifold_extra = hopf_manifold[len(A_train):]

fig, ax = plt.subplots()
ax.plot(A_train, hopf_manifold_train, color='r', linestyle='-')
ax.plot(A_extra, hopf_manifold_extra, color='r', linestyle=':')
ax.plot(A_values, 1 + A_values**2, color='b')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.legend(['Trendy Train', 'Trendy Extrap.', r'True : $B = 1 + A^2$'])
ax.set_title('Estimated Hopf Manifold')
plt.savefig('./figs/est_hopf_manifold.png')
plt.close()

## Plot the real and imaginary parts of the eigenvalues
#print(f'Hopf bifurcation detected at B={hopf_candidates[0]}')
#print('All Hopf candidates:')
#print(hopf_candidates)
#plt.figure(figsize=(10, 6))
#
## Plot real parts
#plt.plot(B_values, real_parts[:, 0], label='Real Part of Eigenvalue 1', color='red', linestyle='-')
#plt.plot(B_values, real_parts[:, 1], label='Real Part of Eigenvalue 2', color='blue', linestyle='-')
#
## Plot imaginary parts
#plt.plot(B_values, imag_parts[:, 0], label='Imaginary Part of Eigenvalue 1', color='red', linestyle='--')
#plt.plot(B_values, imag_parts[:, 1], label='Imaginary Part of Eigenvalue 2', color='blue', linestyle='--')
#
#plt.xlabel('x')
#plt.ylabel('Eigenvalues')
#plt.title('Real and Imaginary Parts of the Eigenvalues of the Jacobians')
#plt.legend()
#plt.grid(True)
#
## Ensure the directory exists
#os.makedirs('./figs/', exist_ok=True)
#
## Save the plot
#plt.savefig('./figs/eigenvalues_real_and_imag_parts.png')
#print("Plot saved to ./figs/eigenvalues_real_and_imag_parts.png", flush=True)

