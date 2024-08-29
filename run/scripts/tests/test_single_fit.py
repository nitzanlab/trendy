import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import numpy as np
import time
from trendy.models import IntegrationScheduler
from trendy.train import set_seed

set_seed(0)

def save_figure(time_points, true_y, pred_y):

    fig, ax = plt.subplots()

    ax.plot(true_y[:,0,0], 'g-', label='True x1')
    ax.plot(true_y[:,0,1], 'b-', label='True x2')

    ax2 = ax.twiny()
    ax2.plot(time_points, pred_y[:,0,0], 'g--', label='Predicted x1')
    ax2.plot(time_points, pred_y[:,0,1], 'b--', label='Predicted x2')
    #ax2.axvline(time_points[current_samples])
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Samples: {num_samples}')
    fn = f'./figs/test_single_fit_dt_{dt:.3f}_{opt_type}.png'
    print(f'Saving figure at {fn}', flush=True)
    plt.savefig(fn)
    plt.close()

# Define the Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, non_autonomous=False):
        super(ODEFunc, self).__init__()
        self.linear1 = nn.Linear(input_dim + (1 if non_autonomous else 0), hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.non_autonomous = non_autonomous
    
    def forward(self, t, x):
        if self.non_autonomous:
            # Concatenate time with the input
            t_tensor = t.expand(x.size(0), 1)  # Ensure t is the correct shape
            x = torch.cat((x, t_tensor), dim=1)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return out

# Neural ODE class
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
    
    def forward(self, x0, t):
        out = torchdiffeq.odeint(self.ode_func, x0, t)
        return out

# Generate some toy time series data
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the true dynamics (example: simple harmonic oscillator)
class TrueDynamics(nn.Module):
    def forward(self, t, x):
        return torch.stack([x[:, 1], -x[:, 0]], dim=-1)

# Data
use_synth_data = False
if use_synth_data:
    num_samples=361
    T = 2*np.pi
    time_points = torch.linspace(0, T, num_samples)
    true_dynamics = TrueDynamics()
    true_y0 = torch.tensor([[1., 0.]])  # Initial condition
    true_y = torchdiffeq.odeint(true_dynamics, true_y0, time_points)
    #subsample 
    step = 361 // 25
    true_y = true_y[::step]
else:
    T = 2*np.pi #30*2*np.pi
    num_samples=2*361 #361 NB: 2 here for 50 gt samples
    time_points = torch.linspace(0, T, num_samples)
    dt = time_points[1] - time_points[0]
    true_y = torch.from_numpy(np.load('./data/min_sample.npy')).unsqueeze(1)
    true_y0 = true_y[0]
    true_y = true_y[:50]

print(f'Using time grid of T={T} with dt={T/num_samples}.')

# Define the model and optimizer
non_autonomous = True
input_dim = 2
hidden_dim = 100
ode_func = ODEFunc(input_dim, hidden_dim, non_autonomous=non_autonomous)
neural_ode = NeuralODE(ode_func).to(device)

opt_type = 'adam'
lr = 1e-3
if opt_type == 'adam':
    optimizer = optim.Adam(neural_ode.parameters(), lr=lr) #.001
else:
    optimizer = optim.SGD(neural_ode.parameters(), lr=lr)
loss_func = nn.MSELoss()

scheduler_type = 'linear'
use_lr_scheduler = False
if use_lr_scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Training loop
num_epochs = 20000
stop_epoch = 15000
num_blocks = 5
min_prop = .5 #.5
loss_history = []
samples_per_epoch = (1-min_prop) * len(true_y) / stop_epoch
#scheduler = IntegrationScheduler('linear', stop_epoch, min_prop=min_prop)
if scheduler_type is not None:
    print(f'Training with {scheduler_type} scheduler')# at {samples_per_epoch:.5f} samples per epoch.')
start = time.time()
current_time_points = time_points
current_true_y = true_y
for epoch in range(0):
    if scheduler_type == 'linear':
        sample_prop = min_prop + (1-min_prop) * min((epoch / stop_epoch), 1.0)
        true_samples = int(sample_prop * len(true_y))
        est_samples  = int(sample_prop * len(time_points))

    elif scheduler_type == 'block':
        block_props = np.linspace(min_prop, 1.0, num_blocks+1)
        epoch_props = np.linspace(0,1.0, num_blocks+1)
        epoch_prop = float(epoch) / num_epochs
        block_ind  = np.where(epoch_prop >= epoch_props)[0][-1]

        sample_prop = block_props[block_ind]
        true_samples = int(sample_prop * len(true_y))
        est_samples  = int(sample_prop * len(time_points))
        
    else:
        sample_prop = 1.0
        true_samples = -1
        est_samples  = -1
 
    optimizer.zero_grad()
    current_time_points = time_points[:est_samples]
    current_true_y = true_y[:true_samples]
    pred_y = neural_ode(true_y0.to(device), current_time_points.to(device))

    # align true and est
    if current_true_y.size(0) > pred_y.size(0):
        step = current_true_y.size(0) // pred_y.size(0)
        pred_aligned = pred_y
        true_aligned = current_true_y[::step]
    else:
        step = pred_y.size(0) // current_true_y.size(0)
        pred_aligned = pred_y[::step]
        true_aligned = current_true_y

    min_length = min(pred_aligned.size(0), true_aligned.size(0))
    pred_aligned = pred_aligned[:min_length]
    true_aligned = true_aligned[:min_length]

    loss = loss_func(pred_aligned, true_aligned)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    print(loss)

    if use_lr_scheduler:
        lr_scheduler.step()

    stop = time.time()

    if epoch % 100 == 0:
        if use_lr_scheduler:
            current_lr = lr_scheduler.get_last_lr()[0]
        else:
            current_lr = lr
        print(f'Epoch {epoch}. Time: {stop-start:.2f}. Loss: {loss.item()}. Using {sample_prop * 100:.3f}% of samples. LR: {current_lr}', flush=True)
        with torch.no_grad():
            pred_y = neural_ode(true_y0.to(device), current_time_points.to(device))
        save_figure(current_time_points, current_true_y, pred_y)
        plt.plot(loss_history)
        plt.yscale('log')
        plt.savefig(f'./figs/loss_history_dt_{dt:.3f}_{opt_type}.png')
        plt.close()

with torch.no_grad():
    pred_y = neural_ode(true_y0.to(device), time_points.to(device))
save_figure(current_time_points, current_true_y, pred_y)
print(true_y[0])
