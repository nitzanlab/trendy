import torch
from torch.utils.data import DataLoader
from trendy import PDE
from trendy.models import get_model, load_checkpoint
from trendy.data import SP2VDataset
from trendy.data._measurements import *
from torchvision import models
from scipy.stats import mode
from scipy.ndimage import correlate
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from time import time
import numpy as np
#plt.style.use('ggplot')
import os
import warnings
import argparse
from joblib import load
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb

# To ignore all warnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='Which model to use, if any. If None, then use true measurement.')
parser.add_argument('--data_dir', required=True,  type=str, help='Which data to load.')
parser.add_argument('--save_dir', type=str, required=True, help='Where to save outputs')
parser.add_argument('--input_dim', default=2, type=int, help='Measurement spapce dim.')
parser.add_argument('--aug_dim', default=4, type=int, help='Parameter spapce dim.')
parser.add_argument('--num_layers', default=4, type=int, help='Number of NODE layers.')
parser.add_argument('--num_features', default=64, type=int, help='Number of NODE features.')
parser.add_argument('--batch_size', default=64, type=int, help='Model batch size.')
args = parser.parse_args()

# Num cpus
num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
#num_cpus = os.cpu_count()
print(f'Using {num_cpus} cpus')
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data preparation
mode = args.data_dir.split('/')[-1]
ds = SP2VDataset(data_dir=args.data_dir)
dl = DataLoader(ds, batch_size=args.batch_size, num_workers=num_cpus, shuffle=False, drop_last= False)

model = get_model('node', args.input_dim, args.aug_dim, args.num_layers, args.num_features).to(device)
model = load_checkpoint(model, args.model_path)
kwargs = {'dt_est': 1e-2, 'T_est': 1, 'device': device}

final_timesteps = []
with torch.set_grad_enabled(False):

    for i, data in enumerate(dl, 0):
        if isinstance(model, torch.nn.DataParallel):
            est = model.model.run(data, **kwargs)
        else:
            est = model.run(data, **kwargs)
        final_timesteps += [state[-1].detach().cpu().numpy() for state in est]

final_timesteps = np.array(final_timesteps)
print(final_timesteps.shape)
np.save(os.path.join(args.save_dir, f'{mode}_final_states.npy'), final_timesteps)
