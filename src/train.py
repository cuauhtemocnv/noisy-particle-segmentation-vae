import torch
from torch.utils.data import DataLoader
from .dataset import NoisySegDataset
from .model import Autoencoder1D
from .utils import weighted_mse


def train_model(epochs=80, batch_size=32, latent_dim=512, lr=1e-3, weight=20.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = NoisySegDataset(n_samples=500)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder1D(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = weighted_mse(out, y, weight=weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss / len(loader):.4f}")

    return model, dataset


if __name__ == "__main__":
    train_model()
