import os
import numpy as np
from collections import Counter

def load_and_count_elements(directory):
    """
    Loads all files labeled k_<i>.npy in the specified directory and prints the
    unique elements and their counts.
    """
    element_counter = Counter()

    for filename in os.listdir(directory):
        if filename.startswith('k_') and filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                data = np.load(file_path)
                flat_data = data.flatten().tolist()
                element_counter.update(flat_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    for element, count in element_counter.items():
        print(f"Element: {element}, Count: {count}")

if __name__ == "__main__":
    directory = './data/scattering_grayscott/train'
    load_and_count_elements(directory)

