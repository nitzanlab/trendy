import torch
import torch.nn as nn
import torch.nn.functional as F
from ._measurements import *
from trendy.train import save_checkpoint
from sklearn.decomposition import PCA
import os

class TRENDy(nn.Module):
    def __init__(self, in_shape, measurement_type='scattering', measurement_kwargs={}, use_log_scale=False, use_pca=True, pca_components=2, num_params=4, node_hidden_layers=[64,64], node_activations='relu', dt=.01, T=1, non_autonomous=False, pca_dir='./models/pca'):
        super(TRENDy, self).__init__()

        self.use_pca          = use_pca
        self.measurement_type = measurement_type
        self.use_log_scale    = use_log_scale
        self.non_autonomous   = non_autonomous

        # Set measurement
        self.set_measurement(measurement_type, in_shape, measurement_kwargs)

        # Compute measurement dim
        with torch.no_grad():
            # Pass random data of shape batch (1) x time (1) * in_shape (c x h x w)
            device = measurement_kwargs.get('device','cpu')
            self.measurement_dim = self.measurement(torch.rand(1, 1, *in_shape).to(device)).shape[-1]

        # Set PCA layer and relevant dimensions
        if use_pca:
            os.makedirs(pca_dir, exist_ok=True)
            self.pca_dir = pca_dir
            self.pca_layer = PCALayer(self.measurement_dim, pca_components)
            self.node_input_dim = pca_components
        else:
            self.node_input_dim = self.measurement_dim

        # Set NODE object
        self.NODE = ParNODE(self.node_input_dim, num_params, hidden_layers=node_hidden_layers, activations=node_activations, dt=dt, T=T, non_autonomous=non_autonomous)

    def set_measurement(self, measurement_type, in_shape, measurement_kwargs):
        '''Set measurement function'''
        self.measurement = get_measurement(measurement_type, in_shape, **measurement_kwargs)

    def compute_measurement(self, U):
        '''Compute measurement of video
           U: tensor shaped batch x time x h x w'''

        # Compute measurement
        measurement = self.measurement(U)

        # Postprocess
        if self.use_log_scale:
            measurement = torch.log10(measurement)

        if self.use_pca:
            measurement = self.pca_layer(measurement)

        return measurement

    def fit_pca(self, dl, pca_timestep=-1, max_samples = np.inf, verbose=True):
        '''Fit PCA and set layer'''

        # Instantiate PCA object from sklearn
        pca = PCA(n_components=self.node_input_dim)


        # Get all states from timestep `pca_timestep`
        print(f'Acquiring states from time {pca_timestep}', flush=True)
        all_states = []
        for d, data in enumerate(dl):
            if verbose:
                print(f'Batch {d}')
            X = data['X']

            # If a video, i.e., not a measurement (batch x time x channel x height x width)
            if X.dim() == 5:
                X = self.measurement(X)

            # Potentially compute PCA on the log of the measurement
            if self.use_log_scale:
                X = torch.log10(X)

            # Get timestep and all states
            X_timestep = X[:,pca_timestep]
            all_states += [xx.numpy() for xx in X_timestep]

            if len(all_states) > max_samples:
                break
        # Fit PCA model
        print(f'Fitting PCA.', flush=True)
        pca.fit(all_states)

        # Set and save PCA layer
        print('Setting and saving pca layer.')
        self.pca_layer.linear.weight.data = torch.tensor(pca.components_, dtype=torch.float32)
        self.pca_layer.mean.data = torch.tensor(pca.mean_, dtype=torch.float32)
        save_checkpoint(self, None, 0, None, save_full_model=False)

    def run(self, init, params):

        init = init.float()
        params = params.float()

        # If initial condition is a video, initialize from measurement (with pca if necessary) of first frame
        if init.dim() == 5:
            # batch x # time x channel x h x w 
            init = self.compute_measurement(init)[:,0]
        # If it's already a measurement, start from there
        elif init.dim() == 2:
            # If alreay a measurement (batch x features)

            # Potentially convert measurement to log
            if self.use_log_scale:
                init = torch.log10(init)

            # If there is a shape mismatch
            if init.shape[-1] != self.node_input_dim:

                # Either it has not yet been reduced by PCA
                if self.use_pca:
                    init = self.pca_layer(init)
                # Otherwise, the measurement is simply not shaped correctly
                else:
                    raise ValueError(f"Initial condition is not correctly shaped. With the measurement '{self.measurement_type} and without pca', the initial condition should have {self.node_input_dim} features.")
        else:
            raise ValueError("Input is incorrectly shaped. It should either be batch x time x channel x height x width or batch x features.")
        return self.NODE.run(init, params)

class PCALayer(nn.Module):
    def __init__(self, in_shape, pca_components):
        super(PCALayer, self).__init__()
        self.mean = nn.Parameter(torch.zeros(in_shape, dtype=torch.float32), requires_grad=False)
        self.linear = nn.Linear(in_shape, pca_components, bias=False)
    
    def forward(self, x):
        x = x - self.mean
        return self.linear(x)
        
class ODEFunc(nn.Module):
    def __init__(self, input_dim, augmented_dim, hidden_layers=[64, 64], activations='relu', non_autonomous=False):
        super(ODEFunc, self).__init__()
        self.augmented_dim = augmented_dim
        total_dim = input_dim + augmented_dim
        self.non_autonomous = non_autonomous
        
        layers = []
        in_features = total_dim + (1 if non_autonomous else 0)

        if activations == 'relu':
            activation_function = torch.nn.ReLU()
        elif activations == 'softplus':
            activation_function = torch.nn.Softplus()
        else:
            raise ValueError('Activation type not recognized. Try relu or softplus.')
       
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation_function)
            in_features = out_features
        
        layers.append(nn.Linear(in_features, total_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, t, z):
        if self.non_autonomous:
            t_tensor = t.expand(z.size(0),1).to(z.device)
            z = torch.cat((z, t_tensor),1)
        return self.model(z)

class ParNODE(nn.Module):
    def __init__(self, input_dim, augmented_dim, hidden_layers=[64, 64], activations='relu', dt=0.01, T=1.0, non_autonomous=False):
        super(ParNODE, self).__init__()

        self.ode_func = ODEFunc(input_dim, augmented_dim, hidden_layers, activations, non_autonomous=non_autonomous)
        self.input_dim = input_dim
        self.set_integration_parameters(dt, T)

    def set_integration_parameters(self, dt, T):
        self.dt = dt
        self.T = T
        
    def run(self, init, params):
        """
        Solves the ODE starting from init (at t=0) and integrates
        up to T, with steps of size dt using a forward Euler method.
        Paramaeters are provided and co-integrated with the state variables. Only the state variables are returned.
        """
        state = torch.cat([init, params], dim=-1)

        # Forward Euler integration
        num_steps = int(self.T / self.dt)
        time_points = torch.linspace(0, self.T, num_steps)
        states = [state]
        for t in time_points:
            state_derivative = self.ode_func(t,state)
            state = state + self.dt * state_derivative
            states.append(state)

        # Stack all states to form the full trajectory
        solution = torch.stack(states, dim=1)

        # Extract only the solution for the original state variables
        return solution[..., :self.input_dim]
