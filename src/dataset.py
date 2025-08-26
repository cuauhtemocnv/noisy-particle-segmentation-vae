import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon
from .utils import add_fourier_noise


class NoisySegDataset(Dataset):
    def __init__(self, n_samples=500, img_size=128, sigma=10.4, low_freq_filter=0.02):
        self.n_samples = n_samples
        self.img_size = img_size
        self.sigma = sigma
        self.low_freq_filter = low_freq_filter
        self.data = []
        self.generate_data()

    def random_triangle(self, img):
        pts = np.random.randint(0, self.img_size, (3, 2))
        rr, cc = polygon(pts[:, 0], pts[:, 1], img.shape)
        img[rr, cc] = 1.0
        return img

    def generate_data(self):
        for _ in range(self.n_samples):
            img = np.zeros((self.img_size, self.img_size))
            # Circle
            r = np.random.randint(8, 20)
            cx, cy = np.random.randint(r, self.img_size - r, 2)
            y, x = np.ogrid[:self.img_size, :self.img_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            img[mask] = 1.0
            # Triangle distractor
            if np.random.rand() < 0.7:
                img = self.random_triangle(img)
            # Noisy
            img_noisy = add_fourier_noise(img, sigma=self.sigma, low_freq_filter=self.low_freq_filter)
            img_noisy = np.clip(img_noisy, 0, 1)
            self.data.append(img_noisy.astype(np.float32))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])[None]
        return x, x  # input = target
