import numpy as np
import torch
from trendy.models import ScatteringPath
from trendy import PDE
from trendy.utils import save_solution_as_movie
import matplotlib.pyplot as plt

def plot_filters(scattering, path, save_fn):
    filter_sequence = scattering.get_filter_sequence(path)
    plt.figure(figsize=(12, 6))

    for i, (wavelet, _) in enumerate(filter_sequence):
        plt.subplot(2, len(filter_sequence), i + 1)
        plt.title(f'Filter {i+1} - Real Part')
        plt.imshow(wavelet.real, cmap='gray')
        plt.axis('off')

        plt.subplot(2, len(filter_sequence), len(filter_sequence) + i + 1)
        plt.title(f'Filter {i+1} - Imaginary Part')
        plt.imshow(wavelet.imag, cmap='gray')
        plt.axis('off')

    plt.savefig(save_fn)
    plt.close()

# Get PDE solution
pde = PDE('GrayScott', params=torch.tensor([.054, .062, .1, .05]))
pde_solution = pde.run().squeeze().detach().numpy()

# Parameters for scattering transform
J = 2
L = 4
max_order = 2
#path = [(0, 0), (1, 1)]  # Example scattering path
index = 49

# Create scattering transform object
scattering = ScatteringPath(J, L, max_order)

path = scattering.linear_index_to_tuple(index)

plot_filters(scattering, path, './figs/filters.png')
