"""Small convolutional VAE for 64x64 frames."""
from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels: int = 2, latent_dim: int = 32):
        super().__init__()
        self.in_channels, self.latent_dim = in_channels, latent_dim
        c = [32, 64, 128, 256]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c[0], 4, 2, 1), nn.ReLU(),   # 32
            nn.Conv2d(c[0], c[1], 4, 2, 1), nn.ReLU(),          # 16
            nn.Conv2d(c[1], c[2], 4, 2, 1), nn.ReLU(),          # 8
            nn.Conv2d(c[2], c[3], 4, 2, 1), nn.ReLU(),          # 4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(c[3] * 16, latent_dim)
        self.fc_logvar = nn.Linear(c[3] * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, c[3] * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c[3], c[2], 4, 2, 1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(c[2], c[1], 4, 2, 1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(c[1], c[0], 4, 2, 1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(c[0], in_channels, 4, 2, 1), nn.Sigmoid(),  # 64
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_dec(z).view(-1, 256, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def encode_mean(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 (B,C,H,W) -> posterior mean (B,D). This is the 'real' latent z_t."""
        return self.encode(to_float(x_uint8))[0]


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float() / 255.0


def vae_loss(recon, x, mu, logvar, beta: float, fg_weight: float = 0.0):
    """Per-image summed squared error + beta * KL, averaged over the batch.

    fg_weight > 0 up-weights pixels that are not background (x > 0.05), so tiny objects such
    as the Pong ball are not ignored by the reconstruction loss.
    """
    sq = (recon - x) ** 2
    if fg_weight > 0:
        sq = sq * (1.0 + fg_weight * (x > 0.05).float())
    rec = sq.sum(dim=(1, 2, 3)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return rec + beta * kl, rec, kl
