import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import TRENDy
from trendy.train import *
from trendy.data import *
from torchvision import models
import matplotlib.pyplot as plt
from time import time
import numpy as np
plt.style.use('ggplot')
import os
import warnings
import json
import argparse
from matplotlib.colors import hsv_to_rgb
from scipy.stats import linregress

def compute_z_scores(data, window_size):
    """
    Compute the Z-scores in a batch of time series with a sliding window.
    
    Parameters:
    - data: numpy array of shape (num_samples, time, features)
    - window_size: int, the size of the sliding window
    
    Returns:
    - z_scores: numpy array of shape (num_samples, new_time, features)
    """
    num_samples, time, features = data.shape
    new_time = time - window_size + 1
    
    # Initialize the output array for Z-scores
    z_scores = np.zeros((num_samples, new_time, features))
    
    for i in range(new_time):
        # Extract the windowed data
        windowed_data = data[:, i:i+window_size, :]
        
        # Compute the mean and std deviation for each window
        local_max = np.max(windowed_data, axis=1)
        local_mean = np.mean(windowed_data, axis=1)
        local_std = np.std(windowed_data, axis=1)

        # Compute the Z-score for the center of the window
        center_index = i + window_size // 2
        #z_scores[:, i, :] = (data[:, center_index, :] - local_mean) / local_std
        z_scores[:, i, :] = (local_max - local_mean) / local_std
    
    return z_scores

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./models/trendy/run_0')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--log_scale', action="store_true")
parser.add_argument('--fig_dir', type=str, default='./figs')
base_args = parser.parse_args()

manifest_fn = os.path.join(base_args.model_dir, 'training_manifest.json')
with open(manifest_fn, 'r') as f:
    training_data = json.load(f)

args = argparse.Namespace()
for key, value in training_data.items():
    setattr(args, key, value)

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
print(f'Using {num_cpus} cpus')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data preparation
train_ds = SP2VDataset(data_dir=os.path.join(args.data_dir, 'train'))
train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=num_cpus, shuffle=False, drop_last=False)
test_ds = SP2VDataset(data_dir=os.path.join(args.data_dir, 'test'))
test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=num_cpus, shuffle=False, drop_last=False)

# Load model
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_log_scale=args.log_scale ,use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est).to(device)
model, _, _ = load_checkpoint(model, base_args.model_dir, device=device)

all_fft = []
all_params = []
for dl in [train_dl, test_dl]:
    for data in dl:
        target = data['X'].to(device)
        params = data['p']
        b,t,f = target.shape
        z = model.pca_layer(target.reshape(-1,f)).reshape(b,t,-1)
        detrended_z = z - z.mean(dim=1,keepdim=True)
        fft_z = torch.fft.fft(detrended_z,dim=1)
        fft_z[0] = 0
        fft_z = torch.fft.fftshift(fft_z)
        power = torch.abs(fft_z)**2
        all_fft += [pp.detach().cpu().numpy() for pp in power]
        all_params += [th.detach().cpu().numpy() for th in params]

N = target.shape[1]
frequencies = torch.fft.fftfreq(N)
frequencies_centered = torch.fft.fftshift(frequencies)[N//2 + 1:-75].numpy()

eps = 1e0
all_fft = np.array(all_fft)[:,N//2 + 1:-75,:] + eps

# Get linear fit in log log
log_freqs = np.log(frequencies_centered)
log_power = np.log(all_fft)
log_mean_power = np.log(all_fft.mean(0))

# Perform a linear fit in log-log space
slope1, intercept1, _, _, _ = linregress(log_freqs, log_mean_power[...,0])
slope2, intercept2, _, _, _ = linregress(log_freqs, log_mean_power[...,1])

# Compute the fitted values and residuals in log-log space
fitted_log_power1 = slope1 * log_freqs + intercept1
fitted_log_power2 = slope2 * log_freqs + intercept2

plt.plot(log_freqs, log_mean_power)
plt.plot(log_freqs, fitted_log_power1)
plt.plot(log_freqs, fitted_log_power2)
plt.legend([r'$S_1$', r'$S_2$', r'$\bar{S}_1$', r'$\bar{S}_2$'])
plt.xlabel('$\omega$')
plt.ylabel('$P(\omega)$')
plt.savefig('./figs/tmp.png')
plt.close()

# Residuals
residuals1 = (log_mean_power[...,0] - fitted_log_power1[None,...])**2
residuals2 = (log_mean_power[...,1] - fitted_log_power2[None,...])**2

max_freqs1 = [frequencies_centered[np.argmax(res)] for res in residuals1]
max_freqs2 = [frequencies_centered[np.argmax(res)] for res in residuals2]

fig, ax = plt.subplots()
ax.plot(log_freqs, residuals1.mean(0))
ax.plot(log_freqs, residuals2.mean(0))
ax.set_xlabel(r'$log(\omega)$')
ax.set_ylabel(r'$r^2$')
ax.legend([r'$S_1$', r'$S_2$'])
fn = os.path.join(base_args.fig_dir, 'min_power_residuals.png')
plt.savefig(fn)
print(f'Saved fig at {fn}.', flush=True)
plt.close()

# Plot power in log log
fig, ax = plt.subplots()
mean = all_fft.mean(0)
std = all_fft.std(0)
ax.plot(frequencies_centered, mean)
for i in range(mean.shape[-1]):
    mu = mean[:,i]
    sig = std[:,i]
    ax.fill_between(frequencies_centered, mu, mu+sig, alpha=.3)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$P(\omega)$')
plt.legend([r'$S_1$', r'$S_2$'])
fn = os.path.join(base_args.fig_dir, 'min_power.png')

plt.savefig(fn)
print(f'Saved figure at {fn}.', flush=True)
plt.close()

#window_size=5
#z_scores = compute_z_scores(all_fft, window_size)
#z_scores = np.nan_to_num(z_scores, nan=0)
#fig, ax = plt.subplots()
#mean_z = z_scores.mean(0)
#std = z_scores.std(0)
#
#window_freqs = np.convolve(frequencies_centered, np.ones(window_size)/window_size, mode='valid')
#ax.plot(window_freqs, mean_z)
#fn = os.path.join(base_args.fig_dir, 'min_power_z_scores.png')
#plt.savefig(fn)
#print(f'Saved figure at {fn}.', flush=True)
#plt.close()

all_params = np.array(all_params)
plt.scatter(all_params[:,0], all_params[:,1], c=max_freqs1, cmap='jet')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Min E level')
plt.xlabel('Min D level')
cbar = plt.colorbar()
cbar.set_label('Est. freq')
fn = os.path.join(base_args.fig_dir, 'min_param_scatter.png')
plt.savefig(fn)
print(f'Saved figured at {fn}.')
plt.close()
