"""Checkpoint loading helpers."""
from __future__ import annotations

import torch

from models.dynamics import LatentDynamics
from models.vae import VAE
from utils import ckpt_path


def load_vae(cfg: dict, tag: str, device) -> VAE:
    ck = torch.load(ckpt_path(cfg, tag, "vae.pt"), map_location=device)
    vae = VAE(**ck["init"])
    vae.load_state_dict(ck["state"])
    return vae.to(device).eval()


def load_dynamics(cfg: dict, tag: str, device) -> LatentDynamics:
    ck = torch.load(ckpt_path(cfg, tag, "dynamics.pt"), map_location=device)
    dyn = LatentDynamics(**ck["init"])
    dyn.load_state_dict(ck["state"])
    return dyn.to(device).eval()
