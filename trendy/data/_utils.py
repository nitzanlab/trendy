import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.special import comb
import pickle
import glob

def set_reps(terms, coefficients, library_terms, coeff_rep, eq_rep, channel):
    """
    Update the representations for polynomial and equation forms.

    Args:
    terms (list): List of terms to be represented.
    coefficients (list): Corresponding coefficients for each term.
    library_terms (list): List of all possible terms in the library.
    coeff_rep (torch.Tensor): Tensor to store the coefficients.
    eq_rep (np.ndarray): Array to store the equation representation.
    channel (int): The channel (column) in the representations to update.
    """
    for term, coeff in zip(terms, coefficients):
        try:
            idx = library_terms.index(term)
        except ValueError:
            raise ValueError(f"Term '{term}' not found in library_terms.")
 
        coeff_rep[idx, channel] = coeff
        eq_rep[idx, channel] = term

def get_library_terms(c, poly_order, der_order, rd=False):
    """
    Report the names of terms in the library`

    """
    
    # All variable names and spatial derivatives
    var_names = ['u', 'v']
    der_names = ['u_x', 'v_x', 'u_y', 'v_y', 'u_{xx}', 'v_{xx}', 'u_{xy}', 'v_{xy}', 'u_{yy}', 'v_{yy}']
   
    # Prune unneeded terms 
    if rd:
        #der_names = der_names[4:]
        der_names = ['L']
    else:
        if c  < 2:
            var_names = [nm for nm in var_names if 'v' not in nm]
            der_names = [nm for nm in der_names if 'v' not in nm]
            if der_order == 0:
                var_names = var_names[0]
                der_names = []
            elif der_order == 1:
                    der_names = der_names[:2]
        else:
            if der_order == 0:
                der_names = []
            elif der_order == 1:
                der_names = der_names[:4]

    library_terms = []
    poly_terms    = []

    poly_terms += var_names
    
    # quadratic terms
    if poly_order > 1:
        for i in range(c):
            for j in range(i,c):
                if i != j:
                    poly_terms.append(var_names[i] + var_names[j])
                else:
                    poly_terms.append(var_names[i] + '^2')

    # cubic terms
    if poly_order > 2:
        for i in range(c):
            for j in range(i,c):
                for k in range(j,c):
                    if i == j == k:
                        poly_terms.append(var_names[i] + '^3')
                    elif i != j and j == k:
                        poly_terms.append(var_names[i] + var_names[j] + '^2')
                    elif i == j and j != k:
                        poly_terms.append(var_names[i] +'^2' + var_names[k])
                    elif i==k and i!=j:
                        poly_terms.append(var_names[i] + '^2' + var_names[j])                        
                    else:
                        poly_terms.append(var_names[i] + var_names[j] + var_names[k])
                        
    # Dictionary also includes polynomials times one spatial derivative
    mixed_terms = []   
    if not rd:
        for d_name in der_names:
            for p_term in poly_terms:
                mixed_terms.append(p_term + d_name)
    library_terms = ['1'] + poly_terms + der_names + mixed_terms
        
    return library_terms
    
def total_derivative_terms(d, m):
    """Calculate the total number of distinct spatial derivative terms 
    up to order m in d dimensions, including mixed partials."""
    total = 0
    for k in range(1, m + 1):
        total += comb(d + k - 1, k)
    return int(total)
    
def total_polynomial_terms(d, n):
    """Calculate the total number of polynomial terms up to degree n in d dimensions."""
    return int(comb(d + n, n))

def total_terms_in_multi_channel_pde(n, d, m, c):
    """Calculate the total number of terms in a multi-channel d-dimensional PDE."""
    D_dm = total_derivative_terms(d, m)
    polynomial_terms = total_polynomial_terms(d, n)
    return c * polynomial_terms * (c * D_dm + 1)

def add_channel(tensor, mode='zero'):
    """
    Adds a new channel to a c x n x n tensor.
    
    Parameters:
    tensor (torch.Tensor): A c x n x n tensor.
    mode (str or int): If 'zero', adds a zero channel. If int, duplicates the specified channel index.
    
    Returns:
    torch.Tensor: A (c+1) x n x n tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if len(tensor.shape) != 3:
        raise ValueError("Tensor must have 3 dimensions (c x n x n).")
    
    c, n, n = tensor.shape
    if isinstance(mode, str) and mode == 'zero':
        new_channel = torch.zeros(1, n, n, dtype=tensor.dtype, device=tensor.device)
    elif isinstance(mode, int):
        if mode < 0 or mode >= c:
            raise ValueError("Channel index out of range.")
        new_channel = tensor[mode, :, :].unsqueeze(0)
    else:
        raise ValueError("Mode must be 'zero' or an integer specifying the channel index.")

    return torch.cat([tensor, new_channel], dim=0)

def compute_kinetic_energy(solution):
    """
    Computes the kinetic energy of the solution at each time step.

    Args:
    - solution (torch.Tensor): Solution tensor of shape [t, c, n, n].

    Returns:
    - np.ndarray: Kinetic energy over time.
    """
    # Ensure the solution is detached and on the CPU
    solution_np = solution.detach().cpu().numpy()

    # Kinetic energy: 0.5 * sum((du/dt)^2) over spatial dimensions
    kinetic_energy = 0.5 * np.sum(np.diff(solution_np, axis=0)**2, axis=(1, 2, 3))

    return kinetic_energy

class SP2VDataset(Dataset):
    def __init__(self,data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(glob.glob(f'{self.data_dir}/X_*.pt'))

    def __getshape__(self,idx):
        x_path = os.path.join(self.data_dir, f'X_0.pt')
        x = torch.load(x_path)
        return x.shape

    def __getitem__(self,idx):
        # Load the data sample on demand
        x_path = os.path.join(self.data_dir, f'X_{idx}.pt')
        p_path = os.path.join(self.data_dir, f'p_{idx}.pt')
        y_path = os.path.join(self.data_dir, f'y_{idx}.pt')

        X = torch.load(x_path)
        p = torch.load(p_path)
        try:
            y = torch.load(y_path)
        except:
            y = torch.tensor(0)

        if self.transforms is not None:
            X = self.transforms(X)

        return {'X' : X, 'p' : p, 'y' : y}

class LogScaleTransform:
    def __call__(self, batch):
        return torch.log(batch)

class UnitNormalize(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

class SpatialResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        # Get the shape of the input tensor
        *other_dims, h, w = tensor.shape
        
        # Reshape the tensor to [..., h, w]
        reshaped_tensor = tensor.view(-1, h, w)
        
        # Apply resizing
        resized_tensor = F.interpolate(reshaped_tensor.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False)
        
        # Reshape back to the original shape
        resized_tensor = resized_tensor.squeeze(0).view(*other_dims, *self.size)
        
        return resized_tensor

def truncate_float(f, decimal_places=4):
    format_string = f"{{:.{decimal_places}f}}"
    return format_string.format(f)

def populate_coefficients(array, tensor):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if tensor[i, j] != 0:
                truncated_coefficient = truncate_float(tensor[i, j].item())
                array[i, j] = truncated_coefficient + array[i, j]

def format_equations(array):
    equations = []
    for col in range(array.shape[1]):
        terms = []
        for term in array[:, col]:
            if term:
                if terms and not term.startswith('-'):
                    terms.append('+')
                terms.append(term)
        equation = ' '.join(terms)
        equations.append(equation)
    return equations

class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def compute_power(tensor, log_plus=False, zero_dc=True):
    nx = tensor.shape[-1]
    omega = torch.fft.fftshift(torch.fft.fft2(tensor))
    if zero_dc:
        omega[..., nx//2,nx//2] = 0
    if log_plus:
        omega = torch.log(1 + torch.abs(omega))
    else:
        omega = torch.abs(omega)
    return omega

def channel_pad(tensor):
    return torch.cat([tensor, tensor[:,:,-1].unsqueeze(1)], dim=2)

def brusselator_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[2]] if rd else [params[2], params[2]]
    nonzero_terms_u = ['1', 'u', 'u^2v'] + spatial_ders
    eq_coeffs_u = [params[0], -(params[1] + 1), 1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[3]] if rd else [params[3], params[3]]
    nonzero_terms_v = ['u', 'u^2v'] + spatial_ders
    eq_coeffs_v = [params[1], -1] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def activatorSD_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[1]] if rd else [params[1], params[1]]
    nonzero_terms_u = ['u', 'u^2v'] + spatial_ders
    eq_coeffs_u = [-1, 1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[2]] if rd else [params[2] , params[2]]
    nonzero_terms_v = ['1', 'u^2v'] + spatial_ders
    eq_coeffs_v = [params[0], -1 * params[0]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep




def fitzhughNagumo_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[4]] if rd else [params[4], params[4]]
    nonzero_terms_u = ['1', 'u', 'v', 'u^3'] + spatial_ders
    # params[0] = lambda, params[1] = sigma, params[2] = kappa
    eq_coeffs_u = [-1 * params[2], params[0], -1 * params[1], -1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[5] / params[3]] if rd else [params[5] / params[3], params[5] / params[3]]
    nonzero_terms_v = ['u', 'v'] + spatial_ders
    eq_coeffs_v = [1. / params[3], -1. / params[3]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def newFN_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    # Diffusion coefficient for u is missing in params, assuming a placeholder diffusion_du
    diffusion_du = kwargs.get('diffusion_du', 1.0)
    spatial_pars = [diffusion_du] if rd else [diffusion_du, diffusion_du]
    nonzero_terms_u = ['1', 'u', 'v', 'u^2', 'u^3'] + spatial_ders
    eq_coeffs_u = [params[3], -1, -1, params[0], -1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    # Diffusion coefficient for v is missing in params, assuming a placeholder diffusion_dv
    diffusion_dv = kwargs.get('diffusion_dv', 1.0)
    spatial_pars = [diffusion_dv / params[2]] if rd else [diffusion_dv / params[2], diffusion_dv / params[2]]
    nonzero_terms_v = ['u', 'v'] + spatial_ders
    eq_coeffs_v = [1. / params[2], -params[1] / params[2]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def simpleFN_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[1]] if rd else [params[1], params[1]]
    nonzero_terms_u = ['u', 'v', 'u^3'] + spatial_ders
    # params[0] = lambda, params[1] = sigma, params[2] = kappa
    eq_coeffs_u = [1, -1, -1./3] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[2] / params[0]] if rd else [params[2] / params[0], params[2] / params[0]]
    nonzero_terms_v = ['u', 'v'] + spatial_ders
    eq_coeffs_v = [1. / params[0], -1. / params[0]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def oregonator_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[2]] if rd else [params[2], params[2]]
    nonzero_terms_u = ['1', 'u', 'uv', 'u^2'] + spatial_ders
    eq_coeffs_u = [-1 * params[0], params[0] - 1, -1 * params[1], 1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[3] / params[1]] if rd else [params[3] / params[1], params[3] / params[1]]
    nonzero_terms_v = ['uv', 'v'] + spatial_ders
    eq_coeffs_v = [params[1], -1 * params[1]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def grayScott_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[2]] if rd else [params[2], params[2]]
    nonzero_terms_u = ['1', 'u', 'uv^2'] + spatial_ders
    eq_coeffs_u = [params[0], -1 * params[0], -1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[3]] if rd else [params[3], params[3]]
    nonzero_terms_v = ['v', 'uv^2'] + spatial_ders
    eq_coeffs_v = [-1 * (params[0] + params[1]), 1] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def lotkaVolterra_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[1]] if rd else [params[1], params[1]]
    nonzero_terms_u = ['u', 'uv'] + spatial_ders
    eq_coeffs_u = [1, -1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[2]] if rd else [params[2], params[2]]
    nonzero_terms_v = ['v', 'uv'] + spatial_ders
    eq_coeffs_v = [-1 * params[0], params[0]] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def thomas_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    spatial_ders = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars = [params[2]] if rd else [params[2], params[2]]
    nonzero_terms_u = ['1', 'u', 'u^3', 'u^2v'] + spatial_ders
    eq_coeffs_u = [-1 * params[0], -1, -4, 1] + spatial_pars
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    spatial_ders = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars = [params[3]] if rd else [params[3], params[3]]
    nonzero_terms_v = ['1', 'v', 'u^2v'] + spatial_ders
    eq_coeffs_v = [-1 * params[1], -1, 1] + spatial_pars
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def polynomial_dictionary_map(params, library_terms, **kwargs):
    # Extract optional arguments with defaults
    device = kwargs.get('device', 'cpu')

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # du
    nonzero_terms_u = library_terms
    eq_coeffs_u = params[:len(params) // 2]
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # dv
    nonzero_terms_v = library_terms
    eq_coeffs_v = params[len(params) // 2:]
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def barkley_dictionary_map(params, library_terms, **kwargs):
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    a, b, Du, Dv = params[0], params[1], params[2], params[3]

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # Activator (u)
    spatial_ders_u = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars_u = [Du] if rd else [Du, Du]
    nonzero_terms_u = ['u', 'u^3', 'uv'] + spatial_ders_u
    eq_coeffs_u = [1, -1, -1 * b] + spatial_pars_u
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # Inhibitor (v)
    spatial_ders_v = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars_v = [Dv] if rd else [Dv, Dv]
    nonzero_terms_v = ['u', 'v'] + spatial_ders_v
    eq_coeffs_v = [a, -1] + spatial_pars_v
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

def schnakenberg_dictionary_map(params, library_terms, **kwargs):
    device = kwargs.get('device', 'cpu')
    rd = kwargs.get('rd', True)

    a, b, Du, Dv = params[0], params[1], params[2], params[3]

    poly_rep = torch.zeros((len(library_terms), 2)).to(device)
    eq_rep = np.full((len(library_terms), 2), '', dtype=object)

    # First chemical (u)
    spatial_ders_u = ['L'] if rd else ['u_{xx}', 'u_{yy}']
    spatial_pars_u = [Du] if rd else [Du, Du]
    nonzero_terms_u = ['1', 'u^2v', 'u'] + spatial_ders_u
    eq_coeffs_u = [a, -1, -1 * b] + spatial_pars_u
    set_reps(nonzero_terms_u, eq_coeffs_u, library_terms, poly_rep, eq_rep, 0)

    # Second chemical (v)
    spatial_ders_v = ['L'] if rd else ['v_{xx}', 'v_{yy}']
    spatial_pars_v = [Dv] if rd else [Dv, Dv]
    nonzero_terms_v = ['u^2v', 'v'] + spatial_ders_v
    eq_coeffs_v = [b, -1] + spatial_pars_v
    set_reps(nonzero_terms_v, eq_coeffs_v, library_terms, poly_rep, eq_rep, 1)

    return poly_rep, eq_rep

