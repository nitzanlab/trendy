import matplotlib.pyplot as plt
import imageio
import os
import torch
import numpy as np
import tempfile
import shutil
from matplotlib.ticker import ScalarFormatter
from io import BytesIO
from PIL import Image

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

    print(f"Movie saved as {file_path}", flush=True)
    plt.close()


# Function to convert a Matplotlib figure to a TensorBoard image
def plot_to_tensorboard(figure, writer, tag, step):
    # Convert figure to image
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)

    # Log image to TensorBoard
    writer.add_image(tag, image, step, dataformats='HWC')
    buf.close()
