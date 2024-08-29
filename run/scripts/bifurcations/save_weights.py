import torch
from trendy.models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./models/trendy')
parser.add_argument('--run_name', type=str, default='run')
base_args = parser.parse_args()

with open(os.path.join(base_args.model_dir, base_args.run_name, 'training_manifest.json'), 'r') as file:
    args_dict = json.load(file)

# Convert the dictionary to an argparse.Namespace object
args = argparse.Namespace(**args_dict)

# Set up model
model = TRENDy(args.in_shape, measurement_type=args.measurement_type, use_pca=args.use_pca, pca_components=args.pca_components, num_params=args.num_params, node_hidden_layers=args.node_hidden_layers, node_activations=args.node_activations, dt=args.dt_est, T=args.T_est).to(device)
model.load_state_dict(torch.load(os.path.join(base_args.model_dir, base_args.run_name, 'model.pt')))

f = model.NODE.ode_func
numpy_save_weights(f, args.save_path)

