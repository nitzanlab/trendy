import json
import torch
import numpy as np
import torchdiffeq
from typing import Union
from ._utils import * 
from ._solve import *
import time
from torch.autograd.functional import jacobian

# Load PDE configurations from a JSON file
with open('../trendy/data/pde_configurations.json', 'r') as f:
    pde_configurations = json.load(f)

# Map PDE data names to their dictionary_map functions
dictionary_map_functions = {
    "ActivatorSD": activatorSD_dictionary_map,
    "GrayScott": grayScott_dictionary_map,
    "GrayScottFull": grayScott_dictionary_map,
    "FitzHughNagumo": fitzhughNagumo_dictionary_map,
    "SimpleFN": simpleFN_dictionary_map,
    "NewFN": newFN_dictionary_map,
    "LotkaVolterra": lotkaVolterra_dictionary_map,
    "Brusselator": brusselator_dictionary_map,
    "BrusselatorFull": brusselator_dictionary_map,
    "Polynomial": polynomial_dictionary_map,
    "Schnakenberg": schnakenberg_dictionary_map,
    "Barkley": barkley_dictionary_map,
    "Thomas": thomas_dictionary_map,
    "Oregonator": oregonator_dictionary_map
}

class PDE(torch.nn.Module):
    def __init__(self, pde_name: str, params=None, device='cpu', use_dictionary_form=False, grad_mask=None,  **kwargs):
        super(PDE, self).__init__()
        if pde_name not in pde_configurations:
            raise ValueError(f"PDE name {pde_name} is not in configurations.")
        self.config = pde_configurations[pde_name]

        # Update config with kwargs
        self.config.update(kwargs)
        preshaped = self.config.get('preshaped', False)
        self.poly_order = self.config.get('poly_order', 3)
        self.der_order = self.config.get('der_order', 2)
        self.channels = self.config.get('channels', 2)
        self.dim = self.config.get('dim', 2)
        self.rd = self.config.get('rd', True)
        self.device=device

        self.dictionary_map = dictionary_map_functions[pde_name]

        if params is None:
            param_ranges = self.config["recommended_param_ranges"]
            if pde_name == 'Polynomial':
                num_terms = total_polynomial_terms(self.dim, self.poly_order) * self.channels + self.channels
                param_ranges = param_ranges * num_terms
            self.flat_params = torch.stack([torch.rand(1) * (high - low) + low for low, high in param_ranges])
        else:
            self.flat_params = params

        # Make sure diffusion parameters are non-negative
        self.flat_params[-self.dim:] = torch.clamp(self.flat_params[-self.channels:],min=0)

        # Format parameters
        library_terms = get_library_terms(2,self.poly_order, self.der_order, self.rd)
        self.library_terms = library_terms
        dictionary_form, eq_rep = self.dictionary_map(self.flat_params, library_terms, rd=self.rd, channels=self.channels, device=device)
        self.eq_rep = eq_rep

        if pde_name == 'Brusselator':
            a, b = self.flat_params[0], self.flat_params[1]
            self.config['solver_params']['center'] = [a, b/a]
       
        # Make trainable
        if use_dictionary_form:
            self.params = torch.nn.ParameterList([torch.nn.Parameter(p) for p in dictionary_form.T.reshape(-1)])
            self.dictionary_map = dictionary_map_functions['Polynomial']
        else:
            self.params = torch.nn.ParameterList([torch.nn.Parameter(p) for p in self.flat_params])
        if grad_mask is not None:
            for p, par in enumerate(self.params):
                if grad_mask[p] == 1:
                    self.params[p].requires_grad = False

    def get_diffusion_params(self):
        return torch.stack([p for p in self.params])[-self.channels:]

    def run(self, solver_params: dict = None, init: torch.Tensor = None, num_inits: int = 1, use_adjoint: bool = False, verbose: int = 1, train=False, new_solver=False) -> torch.Tensor:
        if solver_params is None:
            solver_params = self.config['solver_params']

        # Extract relevant parameters
        T = solver_params['T']
        dt = solver_params.get('dt', 0.01)
        dx = solver_params.get('dx', 1.0)
        solver_name = solver_params.get('solver', 'euler')

        if verbose > 0: 
            diffusion_params = self.get_diffusion_params()
            cfl_check(dt, dx, diffusion_params)

        # Prepare initial conditions
        init = initial_condition(solver_params, num_inits=num_inits) if init is None else init
        # Define time points for the solution
        time_points = torch.linspace(0, T, int(T / dt) + 1)

        # Select the ODE integration method
        if use_adjoint:
            if not any(self.parameters()):
                raise ValueError("Adjoint method requires at least one learnable parameter in PDEFamily.")
            solver = torchdiffeq.odeint_adjoint
        else:
            solver = torchdiffeq.odeint

        if train:
            context = DummyContextManager()
            self.train()
        else:
            context = torch.no_grad()
            self.eval()

        with context:
            # Perform the integration using torchdiffeq with the specified solver
            if new_solver:
                solution = forward_euler(self.forward, init, dt, len(time_points))
            else:
                solution = solver(self, init, time_points, method=solver_name)

        if solution.mean() != solution.mean():
            return None
        else:
            return solution

    def forward(self, t, z):
        return self.compute_derivative(z)

    def compute_derivative(self, z):
        # Extract solver parameters from config
        dx = self.config['solver_params'].get('dx', 1.0)

        reshaped_params, _ = self.dictionary_map(self.params, self.library_terms, rd=self.rd, channels=self.channels, device=self.device)

        # Move dimensions to match evaluate_library's expected input format
        z = z.movedim(-3, -1) # move channels to the end

        # Compute the derivative based on the reaction-diffusion flag
        if self.rd:
            reaction_params = reshaped_params[:-1, :]
            diffusion_params = reshaped_params[-1, :]
            R = reaction(z, reaction_params, self.poly_order, self.der_order, rd=self.rd, dx=dx)
            D = diffusion(z, diffusion_params, dx)
            zdot = R + D
        else:
            basis = evaluate_library(z, self.poly_order, self.der_order, rd=self.rd, dx=dx)
            zdot = torch.einsum('bhwl,ld->bhwd', basis, reshaped_params)

        # Move dimensions back to original format
        zdot = zdot.movedim(-1,-3)
        return zdot
    
    def get_equations(self):

        reshaped_params, _ = self.dictionary_map(self.params, self.library_terms, rd=self.rd, channels=self.channels, device=self.device)

        eq_rep = self.eq_rep
        populate_coefficients(eq_rep, reshaped_params)
        
        eq_strings = format_equations(eq_rep)

        return eq_strings

    def reaction_jacobian(self, z, differentiable=False, return_R=False):
        dx = self.config['solver_params'].get('dx', 1.0)
       
        reshaped_params, _ = self.dictionary_map(self.params, self.library_terms, rd=self.rd, channels=self.channels, device=self.device)
       
        reaction_params = reshaped_params[:-1, :]

        def wrapper_reaction(z):
            if len(z.shape) < 4:
                z = z.reshape(1,1,1,self.channels)
            R = reaction(z, reaction_params, self.poly_order, self.der_order, rd=self.rd, dx=dx).view(-1)
            return R

        if return_R:
            return jacobian(wrapper_reaction, z, create_graph=differentiable), wrapper_reaction(z)
        else:
            return jacobian(wrapper_reaction, z, create_graph=differentiable)

    def dispersion_relation(self, k, differentiable=False, z=None, return_Jr=False):
        dx = self.config['solver_params'].get('dx', 1.0)

        reshaped_params, _ = self.dictionary_map(self.params, self.library_terms, rd=self.rd, channels=self.channels, device=self.device)
        Du, Dv = reshaped_params[-1, :]

        if z is None:
            raise ValueError("TODO: Fixed point solver")
        z.requires_grad = differentiable

        # Reaction term and its Jacobian aat fixed point
        Jr, R = self.reaction_jacobian(z, differentiable=differentiable, return_R=True)

        # Jacobian of diffusion term for all wave numbers
        Jd = torch.stack([torch.diag(torch.tensor([-Du * kk**2, -Dv * kk**2])) for kk in k])

        # Jacobian of full equation
        Jrd = Jr.unsqueeze(0) + Jd

        # Get determinant of Jdr
        detJrd = torch.det(Jrd)

        # Spectrum for RD equation
        evals = torch.linalg.eigvals(Jrd)

        if return_Jr:
            return evals, Jr, R, detJrd
        else:
            return evals
