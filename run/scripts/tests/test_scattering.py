import torch
from kymatio.torch import Scattering2D

# Define parameters
J = 2
L = 8
max_order = 2
num_channels = 2

# Create a dummy image
height, width = 64, 64  # Define a height and width for the test
x = torch.randn(1, num_channels, height, width)

# Create a Scattering2D object
scattering = Scattering2D(J=J, shape=(height, width), L=L, max_order=max_order)

# Apply scattering transform
Sx = scattering(x)
print(Sx.shape)

# Check the shape of the output
Sx_shape = Sx.shape

# Flatten the output to simulate the pytorch flattening
Sx_flattened = Sx.view(Sx.shape[0], -1)

# Calculate the number of dimensions
num_dimensions = Sx_flattened.shape[1]

print(f"Number of dimensions: {num_dimensions}")
