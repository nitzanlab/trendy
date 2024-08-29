import torch
from ._models import ParNODE, ODEFunc
import json
import numpy as np

def save_weights(model, filepath):
    torch.save(model.state_dict(), filepath)

def numpy_save_weights(model, filepath):
    """
    Saves all model weights to a single .npz file with float64 precision.
    Args:
    model (torch.nn.Module): The PyTorch model whose weights are to be saved.
    filepath (str): Path to the .npz file where weights will be saved.
    """
    params = {}
    # Iterate through all model parameters
    for name, param in model.named_parameters():
        # Convert parameter to numpy array with float64 precision and store in dict
        params[name] = param.data.numpy().astype(np.float64)

    # Save all parameters in one .npz file
    np.savez(filepath, **params)
    print(f"All model weights have been saved to {filepath}")

class IntegrationScheduler(object):
    def __init__(self, scheduler_type, max_samples, stop_epoch, min_prop=.1):
        self.min_prop     = min_prop
        self.current_prop = min_prop
        self.max_samples  = max_samples
        self.stop_epoch   = stop_epoch
        
        if scheduler_type == 'linear':
            self.update = self.linear_update
        else:
            raise ValueError('Scheduler type not recognized.')

    def update_model(self, model):
        model_core = model.module if isinstance(model, torch.nn.DataParallel) else model
        current_samples = int(self.current_prop * self.max_samples)
        dt = model_core.NODE.dt
        new_T = dt * current_samples
        model_core.NODE.set_integration_parameters(dt,new_T)

    def linear_update(self, model, epoch):
        self.current_prop = self.min_prop + (1.0 - self.min_prop) * min((float(epoch) / self.stop_epoch, 1.0))
        self.update_model(model)

