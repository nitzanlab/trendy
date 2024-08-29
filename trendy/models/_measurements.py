import pickle
from torchvision import models
import numpy as np
import torch
from joblib import load
from kymatio.torch import Scattering2D
import scipy.signal

class Average(object):
    def __init__(self, in_shape, **kwargs):
        return True
    def measure(self, solution):
        return solution.mean((3,4)).squeeze()

class Scattering(object):
    def __init__(self, in_shape, **kwargs):

        J = kwargs.get('J', 2)
        L = kwargs.get('L',8)
        max_order = kwargs.get('max_order',2)
        device  = kwargs.get('device','cpu')

        self.mini_batch_size = kwargs.get('batch_size', 64)
        self.average = kwargs.get('average', True)

        # Create the scattering transform object with the provided kwargs
        self.scattering_transform = Scattering2D(shape=in_shape[-2:], J=J, L=L, max_order=max_order)
        if device == 'cuda':
            self.scattering_transform = self.scattering_transform.cuda()

    def measure(self, solution):
        # Ensure the input tensor is in the correct format: (batch_size, channels, height, width)
        if solution.dim() != 5:
            raise ValueError("Input solution tensor must be 5D (batch, time, channels, height, width)")

        # Get solution shape
        batch_size, timesteps, channels, height, width = solution.shape

        # Flatten batch and time
        solution = solution.reshape(-1, channels, height, width)

        # Compute scattering batchwise stepping through batch x time
        t = 0
        coefficients = []
        while t < solution.shape[0]:
            effective_batch_size = min(self.mini_batch_size, len(solution) - t)
            batch = solution[t:t+effective_batch_size]

            # Apply the scattering transform
            scattering_coeffs = self.scattering_transform(batch)

            # Average over the spatial dimensions (last two dimensions)
            if self.average:
                scattering_coeffs = scattering_coeffs.mean(dim=(-2, -1)).reshape(effective_batch_size, -1)
            coefficients += [coeff for coeff in scattering_coeffs]
            t += effective_batch_size
        # Reshape output to be b x t x ...
        return torch.stack(coefficients).reshape(batch_size, timesteps, -1)

class ScatteringPath:
    def __init__(self, J, L, max_order):
        self.J = J
        self.L = L
        self.max_order = max_order
        self.wavelets = self.initialize_wavelets()

    def initialize_wavelets(self):
        wavelets = {}
        for j in range(self.J):
            for l in range(self.L):
                wavelets[(j, l)] = self.create_morlet_wavelet(j, l)
        return wavelets

    def create_morlet_wavelet(self, j, l):
        # Create a Morlet wavelet
        frequency = 0.5 ** j
        theta = np.pi * l / self.L
        sigma = 1 / frequency
        n = 21  # Size of the wavelet
        x, y = np.meshgrid(np.arange(-n//2, n//2), np.arange(-n//2, n//2))
        rotx = x * np.cos(theta) + y * np.sin(theta)
        roty = -x * np.sin(theta) + y * np.cos(theta)
        morlet = np.exp(-0.5 * (rotx**2 + roty**2) / sigma**2) * np.exp(1j * 2 * np.pi * frequency * rotx)
        morlet -= morlet.mean()  # Ensure zero mean
        return morlet

    def get_filter_sequence(self, path):
        filter_sequence = []
        for (j, l) in path:
            filter_sequence.append((self.wavelets[(j, l)], 'modulus'))
        return filter_sequence

    def apply_filters(self, image, path):
        if type(path) is int:
            path = self.linear_index_to_tuple(path)
        if image.ndim == 2:  # Single-channel image
            return self._apply_filters_single_channel(image, path)
        elif image.ndim == 3:  # Multi-channel image
            filtered_channels = []
            for c in range(image.shape[2]):
                filtered_channel = self._apply_filters_single_channel(image[:, :, c], path)
                filtered_channels.append(filtered_channel)
            return np.stack(filtered_channels, axis=2)
        else:
            raise ValueError("Unsupported image shape")

    def _apply_filters_single_channel(self, image, path):
        filtered_image = image
        for (wavelet, operation) in self.get_filter_sequence(path):
            filtered_image = scipy.signal.convolve2d(filtered_image, wavelet.real, mode='same')
            if operation == 'modulus':
                filtered_image = np.abs(filtered_image)
        return filtered_image

    def apply_path(self, image, path):
        if isinstance(path, int):
            path = self.linear_index_to_tuple(path)
        return self.apply_filters(image, path)

    def generate_paths(self):
        paths = []

        # Zeroth-order (low-pass) path
        paths.append([])

        # First-order paths
        for j1 in range(self.J):
            for l1 in range(self.L):
                paths.append([(j1, l1)])

        # Second-order paths
        for j1 in range(self.J):
            for l1 in range(self.L):
                for j2 in range(self.J):
                    for l2 in range(self.L):
                        paths.append([(j1, l1), (j2, l2)])

        return paths

    def linear_index_to_tuple(self, index):
        paths = self.generate_paths()
        return paths[index]

    def tuple_to_linear_index(self, path_tuple):
        paths = self.generate_paths()
        return paths.index(path_tuple)

def get_measurement(measurement_type, in_shape, **kwargs):
    if measurement_type == 'average':
        measurement_instance = Average(in_shape, **kwargs)
    elif measurement_type == 'scattering':
        measurement_instance = Scattering(in_shape, **kwargs)
    else:
        raise ValueError('Unknown measurement.')
    return measurement_instance.measure
