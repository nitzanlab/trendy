import matplotlib.pyplot as plt
import imageio
import os
import torch
import numpy as np
import tempfile
import shutil
from trendy.data import compute_kinetic_energy, get_library_terms
from matplotlib.ticker import ScalarFormatter

def save_solution_as_movie(solution, save_dir, file_name='solution_movie.gif', channel=0, step=1):
    """
    Saves the solution as a movie (GIF), using a temporary directory for intermediate files.

    Args:
    - solution (torch.Tensor): Solution tensor of shape [t, c, n, n].
    - save_dir (str): Directory to save the movie.
    - file_name (str): File name for the movie.
    - channel (int): The channel of the solution to visualize.
    - step (int): The plotting period for the video.
    """
    # Ensure the solution tensor is detached and on the CPU
    solution_np = solution.detach().cpu().numpy()
    #mx = solution_np[-1,channel].max()
    #mn= solution_np[-1,channel].min()

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # File path for the movie
    file_path = os.path.join(save_dir, file_name)

    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a list to store the paths of the frames
        frame_paths = []

        # Iterate over each time step and create a frame
        for t in range(0,solution_np.shape[0], step):
            fig, ax = plt.subplots()
            im = ax.imshow(solution_np[t, channel], cmap='viridis')#), vmin=mn, vmax=mx)
            ax.set_title(rf'$t={t}$')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax)
            formatter = ScalarFormatter(useOffset=False)  # Prevents using scientific notation
            cbar.ax.yaxis.set_major_formatter(formatter)
            # Save frame to the temporary directory
            frame_file = os.path.join(temp_dir, f'frame_{t}.png')
            plt.savefig(frame_file)
            frame_paths.append(frame_file)
            plt.close(fig)

        # Read frames from the files and write to a GIF
        frames = [imageio.imread(frame_file) for frame_file in frame_paths]
        imageio.mimsave(file_path, frames, format='GIF', duration=0.1)

    print(f"Movie saved as {file_path}")
    plt.close()

def plot_kinetic_energy(solution, save_dir, file_name='kinetic_energy.png'):
    """
    Plots the kinetic energy of the solution over time.

    Args:
    - solution (torch.Tensor): Solution tensor of shape [t, c, n, n].
    - save_dir (str): Directory to save the plot.
    - file_name (str): File name for the plot.
    """
    kinetic_energy = compute_kinetic_energy(solution)

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Linear scale
    axs[0].plot(kinetic_energy)
    axs[0].set_title('Kinetic Energy Over Time (Linear Scale)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Kinetic Energy')

    # Log scale
    axs[1].plot(kinetic_energy)
    axs[1].set_yscale('log')
    axs[1].set_title('Kinetic Energy Over Time (Log Scale)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Kinetic Energy (Log Scale)')

    # Save the plot
    plt.savefig(f"{save_dir}/{file_name}")
    plt.close(fig)

def plot_parameter_values(data, labels, tick_labels=None, save_dir=None, file_name=None, use_dictionary_form=False):
    """
    Creates a side-by-side bar plot for the given data.

    Parameters:
    - data: List of Numpy arrays, each representing the heights of one set of bars.
    - labels: List of labels for each Numpy array.
    - tick_labels: (Optional) List of strings to replace the x-axis tick labels.
    - save_dir: (Optional) Directory to save the resulting figure.
    - file_name: (Optional) Name of the file to save the figure.
    - use_dictionary_form: (Optional) Whether or not parameters are in dictionary form.
    """
    n_groups = len(data[0])
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.8 / len(data)
    opacity = 0.8

    for i, (dat, label) in enumerate(zip(data, labels)):
        plt.bar(index + i * bar_width, dat, bar_width,
                alpha=opacity, label=label)

    if tick_labels is not None and use_dictionary_form:
        tick_labels = [fr'${tl}$' for tl in tick_labels if tl != '']
        plt.xticks(index + bar_width * (len(data) - 1) / 2, tick_labels)
    else:
        tick_labels = [fr'$\theta_{i}$' for i in range(len(data))]
        plt.xticks(index + bar_width * (len(data) - 1) / 2, tick_labels)

    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.legend()

    plt.tight_layout()

    if save_dir and file_name:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

def compute_radial_transform(image):
    nx = image.shape[0]
    fft2_image = np.fft.fft2(image)
    fft2_shifted = np.fft.fftshift(fft2_image)
    fft2_shifted[nx//2, nx//2] = 0  # Remove DC component
    power_spectrum = np.abs(fft2_shifted)**2
    
    nx, ny = image.shape
    X, Y = np.ogrid[:nx, :ny]
    center_x, center_y = nx // 2, ny // 2
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    R = R.astype(np.int32)
    
    radial_sum = np.bincount(R.ravel(), weights=power_spectrum.ravel())
    radial_count = np.bincount(R.ravel())
    radial_average = radial_sum / radial_count
    
    # Exclude the DC component and the highest frequency
    radial_average = radial_average[1:-1]
    
    return radial_average
