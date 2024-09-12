import os
import matplotlib.patches as mpatches
import argparse
import torch
import matplotlib.pyplot as plt

def load_and_scatter(data_dir, dims, fig_dir='./figs'):
    # Define the modes and their colors
    modes = ['train', 'val', 'test']
    colors = {'train': 'red', 'val': 'green', 'test': 'blue'}
    
    # Initialize a figure for the scatter plot
    plt.figure(figsize=(10,10))

    # Loop through each mode: train, val, test
    for mode in modes:
        mode_dir = os.path.join(data_dir, mode)
        
        # Loop through each p_<i>.pt file in the mode directory
        for file in os.listdir(mode_dir):
            if file.startswith('p_') and file.endswith('.pt'):
                file_path = os.path.join(mode_dir, file)
                
                # Load the tensor
                tensor = torch.load(file_path)
                
                # Ensure the tensor has enough dimensions
                if max(dims) < tensor.size(0):
                    # Scatter the two dimensions
                    plt.scatter(tensor[dims[0]].item(), tensor[dims[1]].item(), color=colors[mode], label=mode)
    
    # Add labels and legend
    plt.xlabel(rf"$A$", fontsize=24)
    plt.ylabel(rf"$B$", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Create custom legend elements
    train_patch = mpatches.Patch(color='red', label='train')
    val_patch = mpatches.Patch(color='green', label='val')
    test_patch = mpatches.Patch(color='blue', label='test')
    
    # Add the legend to the plot
    plt.legend(handles=[train_patch, val_patch, test_patch], fontsize=20)

    # Show the plot
    plt.savefig(os.path.join(args.fig_dir, 'params.png'))
    plt.close()

if __name__ == "__main__":
    # Argument parser to get the data_dir and dimensions from the shell argument
    parser = argparse.ArgumentParser(description="Scatter plot of selected dimensions from p_<i>.pt tensors.")
    parser.add_argument("--data_dir", type=str, help="The path to the data directory")
    parser.add_argument("--fig_dir", type=str, default='./figs', help="The path to the figure directory")
    parser.add_argument("--dims", type=int, nargs='+', default=[0, 1], help="The two dimensions to scatter (default: [0, 1])")
    
    args = parser.parse_args()
    
    # Call the scatter function
    load_and_scatter(args.data_dir, args.dims, fig_dir=args.fig_dir)

