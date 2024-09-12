import torch
from trendy.models import *
from trendy.train import load_checkpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_super_dir', type=str, default='./models/good_models')
base_args = parser.parse_args()

sub_dirs1 = ['no_noise', 'boundaries', 'patches']
sub_dirs2 = ['eps_0.0', 'eps_0.25', 'eps_0.5']
sub_dirs3 = ['pca_2', 'pca_4', 'pca_8']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for sd1 in sub_dirs1:
    for sd2 in sub_dirs2:
        for sd3 in sub_dirs3:
            full_dir = os.path.join(base_args.model_super_dir, sd1, sd2, sd3)

            if not os.path.exists(os.path.join(full_dir, 'checkpoint.pt')):
                print(f'{full_dir} has no checkpoint yet.', flush=True)
                continue

            with open(os.path.join(full_dir, 'training_manifest.json'), 'r') as file:
                args_dict = json.load(file)
            
            # Convert the dictionary to an argparse.Namespace object
            args = argparse.Namespace(**args_dict)
            
            # Set up model
            model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est, non_autonomous=args.non_autonomous, pca_dir=args.pca_dir).to(device)
            
            # Load dir
            model, _, _ = load_checkpoint(model, full_dir, pca_dir=args.pca_dir, device=device)

            # Save path
            save_path = os.path.join(full_dir, 'weights.npz')
            
            f = model.NODE.ode_func
            numpy_save_weights(f, save_path)

