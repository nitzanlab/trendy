import torch
import glob
import os
import numpy as np

data_dir = './data/scattering_min'

# Get running mins and maxes
#for mode in ['train', 'test']:
#    running_min = torch.tensor([torch.inf, torch.inf])
#    running_max = torch.tensor([-1*torch.inf, -1*torch.inf])
#    for fn in glob.glob(os.path.join(data_dir, mode, 'p_*')):
#        param = torch.load(fn)
#        running_min = torch.where(param < running_min, param, running_min)
#        running_max = torch.where(param > running_max, param, running_max)
#print(running_min, flush=True)
#print(running_max, flush=True)

par_min = torch.tensor([1193.3574, 241.8587])
par_max = torch.tensor([22589.8477, 1264.0690])
for mode in ['train', 'test']:
    for fn in glob.glob(os.path.join(data_dir, mode, 'p_*')):
        ind = int(fn.split('_')[-1].split('.')[0])
        param = torch.load(fn)

        new_param = (param - par_min) / (par_max - par_min)


        old_fn = os.path.join(data_dir, mode, f'unnormalized_p_{ind}.pt')
        new_fn = os.path.join(data_dir, mode, f'p_{ind}.pt')

        torch.save(param, old_fn)
        torch.save(new_param, new_fn)
print('Done')
