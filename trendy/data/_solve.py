import torch
import numpy as np
from scipy.optimize import minimize
import warnings
from functools import partial
import torchdiffeq

def evaluate_library(U, poly_order, der_order, rd=False, dx=1.0):
    '''Evaluate dictionary of basis functions on a c-D function, U, on a two-dimensional spatial domain'''
    U = torch.Tensor(U) if isinstance(U, (np.ndarray, np.generic)) else U
    n,h,w,c = U.shape
    
    # Add spatial derivatives
    DU = []
    if not rd: 
        for d in range(der_order):
            if rd and d < 1: continue
            ders = FiniteDiff(U,dx,d + 1)
            for der in ders: # Spatial first, e.g. Ux, Uy
                for i in range(der.shape[-1]): # Then channel, e.g. ux, vx
                    DU.append(der[...,i])
    poly_terms = []
    # First we add terms without spatial derivatives
    # linear terms
    for i in range(c):
        poly_terms.append(U[...,i])
        
    # quadratic terms
    if poly_order > 1:
        for i in range(c):
            for j in range(i,c):
                poly_terms.append(U[...,i] * U[...,j])
    
    if poly_order > 2:
        for i in range(c):
            for j in range(i,c):
                for k in range(j,c):
                    poly_terms.append(U[...,i] * U[...,j] * U[...,k])
    
    # Then we add the spatial derivative terms
    mixed_terms = []   
    if not rd:
        for dU in DU:
            for poly_term in poly_terms:
                mixed_terms.append(poly_term * dU)
    constant = torch.ones(n,h,w,1).to(U.device) # costant term
    poly_terms = torch.stack(poly_terms).movedim(0,-1)
    if not rd:
        DU = torch.stack(DU).movedim(0,-1)
        mixed_terms = torch.stack(mixed_terms).movedim(0,-1)
        library = torch.cat([constant, poly_terms, DU, mixed_terms], axis=-1)
    else:
        library = torch.cat([constant, poly_terms], axis=-1)
    return library
    
def cfl_check(dt: float, dx: float, diffusion_params: torch.Tensor):
    """
    Performs the Courant–Friedrichs–Lewy (CFL) condition check for diffusion.

    Args:
        dt (float): Time step size.
        dx (float): Space step size.
        diffusion_params (torch.Tensor): Diffusion parameters of the system.
    """
    d_max = max(diffusion_params)
    thresh = dx ** 2 / (2 * d_max)
    if dt > thresh:
        warnings.warn('Warning: dt is too large for the diffusion CFL criterion.')

def laplacian(u, dx):
    """
      second order finite differences
      u shape [..., s, s]
    """
    du2 = FiniteDiff(u,dx,2)
    return du2[0] + du2[2]

def forward_euler(f, z0, dt, num_steps):
    zs = [z0]  # List to hold all computed y values
    for i in range(1, num_steps):
        z_next = zs[-1] + dt * f(0,zs[-1])
        zs.append(z_next)
    return torch.stack(zs)  # Convert list to tensor

def reaction(z, params, poly_order, der_order, rd=False, dx=1.0):
    basis = evaluate_library(z, poly_order, der_order, rd=rd, dx=dx)
    return torch.einsum('bhwl,ld->bhwd', basis, params)

def diffusion(z, diffusion_params, dx):
    """z shape [..., s, s, c]"""
    z = z.movedim(-1,0) # [c, ..., s, s]
    
    L = torch.stack([laplacian(species, dx) for species in z]).movedim(0,-1) #[..., s, s, c]
    return torch.cat([diffusion_params[0]*L[...,0].unsqueeze(-1), diffusion_params[1]*L[...,1].unsqueeze(-1)], dim=-1) # [...,s,s,c]

def periodic_bc(grid):
    """
    Apply periodic boundary conditions to a 4D tensor.
    The tensor shape is assumed to be [batch, channel, height, width].
    """
    # Copy top row to bottom and bottom row to top
    grid[:, :, 0, :] = grid[:, :, -2, :]
    grid[:, :, -1, :] = grid[:, :, 1, :]

    # Copy left column to right and right column to left
    grid[:, :, :, 0] = grid[:, :, :, -2]
    grid[:, :, :, -1] = grid[:, :, :, 1]

    return grid

def FiniteDiff(u, dx, d):
    """
    Computes the finite difference derivatives of the input tensor 'u' using the 2nd order central difference method.

    Parameters:
    u (torch.Tensor): The data tensor to be differentiated. Assumes at least 2-dimensional.
    dx (float): Grid spacing. Assumes uniform spacing. Must be non-zero.
    d (int): Order of the derivative. Currently supports 1st and 2nd derivatives (d=1 or d=2).

    Returns:
    list of torch.Tensor: The computed derivatives. For d=1, returns [ux, uy]. For d=2, returns [uxx, uxy, uyy].

    Note:
    - The function assumes periodic boundary conditions.
    - The accuracy decreases for derivatives of order higher than 3.
    """

    if not isinstance(u, torch.Tensor):
        raise TypeError("Input 'u' must be a torch.Tensor.")
    if not isinstance(dx, (float, int)):
        raise TypeError("Input 'dx' must be a float or int.")
    if dx == 0:
        raise ValueError("Input 'dx' must be non-zero.")
    if not isinstance(d, int):
        raise TypeError("Input 'd' must be an integer.")
    if d not in [1, 2]:
        raise ValueError("Derivative order 'd' must be either 1 or 2.")

    if d == 1:
        # Centered finite difference with periodic boundaries for first derivative
        ux_f = torch.roll(u,1,dims=2)
        ux_b = torch.roll(u,-1,dims=2)
        ux = (ux_f - ux_b)  / (2 * dx)

        uy_f = torch.roll(u,1,dims=1)
        uy_b = torch.roll(u,-1,dims=1)
        uy = (uy_f - uy_b)  / (2 * dx)
  
        return [ux, uy]

    if d == 2:
        # Centered finite difference with periodic boundaries for second derivative
        ux_f = torch.roll(u,1,dims=2)
        ux_b = torch.roll(u,-1,dims=2)
        uxx = (ux_f - 2 * u + ux_b) / dx**2

        uy_f = torch.roll(u,1,dims=1)
        uy_b = torch.roll(u,-1,dims=1)
        uyy = (uy_f - 2 * u + uy_b) / dx**2

        u_xy_ff = torch.roll(torch.roll(u,1,dims=2),1,dims=1)
        u_xy_bb = torch.roll(torch.roll(u,-1,dims=2),-1,dims=1)
        u_xy_fb = torch.roll(torch.roll(u,1,dims=2),-1,dims=1)
        u_xy_bf = torch.roll(torch.roll(u,-1,dims=2),1,dims=1)

        uxy = (u_xy_ff - u_xy_fb - u_xy_bf + u_xy_bb) / (4*dx**2)

        return [uxx, uxy, uyy]

def initial_condition(solver_params: dict = None, dim: int = 2, num_inits: int = 1, **kwargs) -> torch.Tensor:
    """
    Generates the initial condition based on specified solver parameters.

    Args:
        solver_params (dict): Parameters dictating the initial condition generation.
        dim (int): Number of channels in the initial condition
        num_inits (int): Number of initial conditions to return

    Returns:
        torch.Tensor: Tensor representing the initial condition.
    """

    init_type = solver_params['init_type']
    nx  = solver_params['nx']
    z = torch.zeros(dim, nx, nx)

    if init_type == 'bump':
        noise_magnitude = solver_params['noise_magnitude']
        z = _init_bump(nx, dim, noise_magnitude, num_inits)
    elif init_type == 'noise':
        noise_magnitude = solver_params['noise_magnitude']
        center = solver_params['center']
        z = torch.cat([u * torch.ones(num_inits, 1, nx, nx) for u in center], dim=1)
        z += torch.randn_like(z) * noise_magnitude
    elif init_type == 'noise_rand_center':
        noise_magnitude = solver_params['noise_magnitude']
        center_range = solver_params['center_range']
        center_u = (torch.rand(1) * (center_range[0][1] - center_range[0][0])) + center_range[0][0]

        center_v = (torch.rand(1) * (center_range[1][1] - center_range[1][0])) + center_range[1][0]
        center = [center_u, center_v]
        z = torch.cat([u * torch.ones(num_inits, 1, nx, nx) for u in center], dim=1)
        z += torch.randn_like(z) * noise_magnitude
    elif init_type == 'brusselator':
        a = solver_params['a']
        b = solver_params['b']
        z = _init_brusselator(a,b, nx, num_inits)
    return periodic_bc(z)

def _init_bump(nx: int, dim: int, noise_magnitude: float, num_inits: int) -> torch.Tensor:
    """
    Initializes the state with a 'bump' pattern and applies noise.

    Args:
        nx (int): Spatial resolution.
        dim (int): Number of channels.
        noise_magnitude (float): Magnitude of the noise to be added.
        num_inits (int): Number of random initializations.

    Returns:
        torch.Tensor: Initialized state tensor with 'bump' pattern.
    """
    z = torch.cat([(i < dim / 2.) * torch.ones((num_inits, 1, nx, nx)) for i in range(dim)], dim=1)
    x, y = torch.meshgrid(torch.linspace(0, 1, nx), torch.linspace(0, 1, nx), indexing='ij')
    mask = (0.4 < x) & (x < 0.6) & (0.4 < y) & (y < 0.6)
    mask = mask.unsqueeze(0).repeat(num_inits,1,1)
    for i in range(dim):
        z[:,i][mask] = 0.25 if i >= int(dim / 2.) else 0.5

    z += noise_magnitude * torch.randn_like(z)
    return torch.clamp(z, 0)

def _init_brusselator(a: float, b: float, nx: float, num_inits: int) -> torch.Tensor:
    """
    Initializes the state using the Brusselator method.

    Args: 
        a (float): Brusselator parameter a.
        b (float): Brusselator parameter b.
        nx (int): Spatial resolution.
        num_inits (int): Number of random initializations.
    Returns:
        torch.Tensor: Initialized state tensor with Brusselator pattern.
    """
    u = a * torch.ones(num_inits, nx, nx)
    v = b / a + 0.1 * torch.randn(num_inits, nx, nx)

    return torch.cat([u.unsqueeze(1),v.unsqueeze(1)],dim=1)

def find_steady_state(initial_z, params, poly_order, der_order, rd, dx):
    # Objective function to minimize
    def objective(z_flat):
        # Reshape z_flat back to its original shape
        z_reshaped = z_flat.reshape(initial_z.shape)
        # Convert z_reshaped to a PyTorch tensor
        z_tensor = torch.tensor(z_reshaped, dtype=torch.float32)
        # Call the PyTorch function
        output_tensor = reaction(z_tensor, params, poly_order, der_order, rd, dx)
        # Convert the output to a NumPy array and calculate the Euclidean norm
        output_np = output_tensor.detach().cpu().numpy()
        norm = np.linalg.norm(output_np)
        print(norm)
        return norm
    
    # Flatten the initial_z to pass to minimize
    initial_z_flat = initial_z.flatten()
    result = minimize(objective, initial_z_flat, tol=1e-3, method='L-BFGS-B')
    
    # Reshape the result back to the original shape
    z_optimal = result.x.reshape(initial_z.shape)
    return torch.tensor(z_optimal).float()
