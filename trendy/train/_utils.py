import torch
import re
import random
import numpy as np
import os
import json
import datetime
from torch.utils.tensorboard import SummaryWriter
import warnings

def initialize_training_environment(args):
    """
    Initialize the training environment.
    
    Args:
    - args: Arguments or configurations for setting up the environment.

    Returns:
    - device: The device to run the model on (CPU or GPU).
    - checkpoint_dir: Directory for saving model checkpoints.
    - writer: TensorBoard writer for logging.
    - num_gpus: Number of available GPUs.
    - num_cpus: Number of CPU cores available for data loading.
    """
    # To ignore all warnings from PyTorch
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    # Detect the number of GPUs
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f'Cuda version: {torch.version.cuda}')
    else:
        print("No GPU available.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory for model checkpoints
    if not args.pretrained:
        run_name = get_next_run_name(args.model_dir)
    else:
        run_name = args.run_name
    checkpoint_dir = os.path.join(args.model_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Tensorboard setup
    full_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(full_log_dir, exist_ok=True)
    writer = SummaryWriter(full_log_dir)

    # Detect the number of CPUs
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))

    print(f'Using {num_cpus} CPUs.')

    return args, device, checkpoint_dir, writer, num_gpus, num_cpus

def set_seed(seed_value=0):
    """Set seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to save a model checkpoint
def save_checkpoint(model, optimizer, epoch, model_dir):
    checkpoint_path = os.path.join(model_dir, f"checkpoint.pt")
    
    # Sometimes you don't have an opt to save or don't want to
    saved_opt = None if optimizer is None else optimizer.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': saved_opt,
    }, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, model_dir, optimizer=None, device='cpu'):
    checkpoint_path = os.path.join(model_dir, f"checkpoint.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']

    # If the model was saved as a DataParallel model, adjust accordingly
    if 'module.' in list(model_state_dict.keys())[0]:
        # Create a new OrderedDict without the 'module' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        model_state_dict = new_state_dict
    
    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to the specified device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    epoch = checkpoint['epoch']
   
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")

    return model.to(device), optimizer, epoch

def get_next_run_name(base_dir):
    # Regular expression to match run_<i> where i is a non-negative integer
    pattern = re.compile(r'run_(\d+)')
    
    # List all subdirectories in the specified directory
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Extract integers from the subdirectory names that match the pattern
    run_indices = []
    for subdir in subdirs:
        match = pattern.match(subdir)
        if match:
            run_indices.append(int(match.group(1)))
    
    # Determine the next integer j
    if run_indices:
        next_run_index = max(run_indices) + 1
    else:
        next_run_index = 0

    # Set the run name
    run_name = f'run_{next_run_index}'
    
    return run_name
