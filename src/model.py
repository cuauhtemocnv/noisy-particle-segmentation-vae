import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Autoencoder1D(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        # Decoder
        self.dec4 = ConvBlock(256, 128)
        self.dec3 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.dec1 = nn.Conv2d(32, 1, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.flatten(self.pool(e4))
        z = self.fc_enc(b)
        return z

    def decode(self, z):
        b = self.fc_dec(z)
        b = b.view(-1, 256, 8, 8)
        d = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec4(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec3(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec2(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.dec1(d))
        return out

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        if return_latent:
            return z
        return self.decode(z)
