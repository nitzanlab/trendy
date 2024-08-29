import numpy as np
import torch
from trendy.models import ScatteringPath
from trendy import PDE
from trendy.utils import save_solution_as_movie

def apply_scattering_to_video(video, scattering_transform, path):
    time, channels, height, width = video.shape
    filtered_video = np.zeros_like(video)
    
    for t in range(time):
        for c in range(channels):
            filtered_video[t, c] = scattering_transform.apply_filters(video[t, c], path)
    
    return filtered_video

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

# Apply scattering transform to the video
filtered_solution = apply_scattering_to_video(pde_solution, scattering, index)

print('Saving filtered solution.', flush=True)
save_solution_as_movie(torch.tensor(filtered_solution), './figs/', file_name='filtered_solution.gif', step=10)
print('Saving true solution.', flush=True)
save_solution_as_movie(torch.tensor(pde_solution), './figs/', file_name='raw_solution.gif', step=10)
