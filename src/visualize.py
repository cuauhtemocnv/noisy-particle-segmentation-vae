import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA

from .dataset import NoisySegDataset
from .model import Autoencoder1D


def visualize_reconstructions(model, dataset, device, n=3):
    model.eval()
    for i in range(n):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().squeeze().numpy()
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(x.cpu().squeeze(), cmap="gray")
        plt.title("Noisy Input")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(y.squeeze(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Reconstruction")
        plt.axis("off")
        plt.show()


def visualize_latent_space(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    latents = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
    latents = np.vstack(latents)
    center = latents.mean(axis=0)

    # PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    center_2d = pca.transform(center.reshape(1, -1))

    plt.figure(figsize=(6, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, label="Latent points")
    plt.scatter(center_2d[0, 0], center_2d[0, 1], c="red", s=100, label="Center of mass")
    plt.legend()
    plt.title("Latent space (2D PCA)")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = NoisySegDataset(n_samples=500)
    model = Autoencoder1D(latent_dim=512).to(device)
    # NOTE: load trained weights here if available
    visualize_reconstructions(model, dataset, device)
    visualize_latent_space(model, dataset, device)
