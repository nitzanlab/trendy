import torch
import os
import glob

data_dir = './data/scattering_brusselator'

for mode in ['train', 'test']:
    mode_dir = os.path.join(data_dir, mode)
    fns = glob.glob(mode_dir + '/X_*')
    for fn in fns:
        X = torch.load(fn)
        torch.save(X.squeeze(), fn)
print('Done')

