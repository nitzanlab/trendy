import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse

def main(args):
    # Load images
    image_files = [f for f in os.listdir(args.data_dir) if f.startswith("U_") and f.endswith(".pt")]
    images = []
    for file in image_files:
        img = torch.load(os.path.join(args.data_dir, file))
        images.append(img.numpy().flatten())  # Flatten the image

    images = np.array(images)  # Convert list of arrays to a 2D array

    # Perform PCA
    pca = PCA(n_components=args.n_components)
    pca_features = pca.fit_transform(images)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=args.n_clusters)
    labels = kmeans.fit_predict(pca_features)

    # Save the labels
    os.makedirs(args.save_dir, exist_ok=True)
    for i, file in enumerate(image_files):
        label_file = f"k_{file.split('_')[1].split('.')[0]}.npy"
        np.save(os.path.join(args.save_dir, label_file), labels[i])

    print("Labels saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform PCA and KMeans clustering on images.")
    parser.add_argument('--n_components', type=int, default=100, help='Number of PCA components.')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of KMeans clusters.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the images.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the labels.')

    args = parser.parse_args()
    main(args)

