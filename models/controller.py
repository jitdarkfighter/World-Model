"""Controllers: random-shooting MPC over imagined rollouts, and the DQN baseline pieces."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from models.dynamics import LatentDynamics
from models.vae import VAE


class MPCController:
    """Random-shooting MPC inside the learned world model.

    At each real step: encode the frame, warm the GRU on the last `context` real (z, a)
    pairs (matches training windows that start from h=0), sample `candidates` random action
    sequences of length `horizon`, imagine them, score each by predicted discounted-free
    return weighted by survival probability, execute the first action of the best one.
    """

    def __init__(self, vae: VAE, dyn: LatentDynamics, action_dim: int, horizon: int,
                 candidates: int, context: int, device, sample: bool = True):
        self.vae, self.dyn, self.A = vae, dyn, action_dim
        self.H, self.N, self.K = horizon, candidates, context
        self.device, self.sample = device, sample
        self.hist: deque = deque(maxlen=context)

    def reset(self) -> None:
        self.hist.clear()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        z = self.vae.encode_mean(x)  # (1,D)
        h = None
        if self.hist:
            zs = torch.cat([p[0] for p in self.hist]).unsqueeze(0)
            acts = torch.tensor([[p[1] for p in self.hist]], device=self.device)
            _, h = self.dyn(zs, acts)
            h = h.expand(-1, self.N, -1).contiguous()
        actions = torch.randint(0, self.A, (self.N, self.H), device=self.device)
        _, rew, done, _ = self.dyn.imagine(z.expand(self.N, -1), actions, h, self.sample)
        # survival before step t = prod_{k<t} (1 - done_k)
        alive = torch.cumprod(torch.cat([torch.ones_like(done[:, :1]), 1 - done[:, :-1]], 1), 1)
        score = (alive * rew).sum(1) + 1e-4 * torch.randn(self.N, device=self.device)  # tie-break
        a = int(actions[score.argmax(), 0])
        self.hist.append((z, a))
        return a


class QNetwork(nn.Module):
    """Standard Atari-style DQN CNN for 64x64 stacked frames."""

    def __init__(self, in_ch: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),   # 15
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),      # 6
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),      # 4
            nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU(), nn.Linear(256, action_dim),
        )

    def forward(self, x):  # x: float tensor with pixel values in [0, 255]
        return self.net(x / 255.0)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape):
        self.cap, self.i, self.size = capacity, 0, 0
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.nobs = np.zeros((capacity, *obs_shape), np.uint8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, o, a, r, no, d):
        self.obs[self.i], self.a[self.i], self.r[self.i] = o, a, r
        self.nobs[self.i], self.d[self.i] = no, d
        self.i = (self.i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, device):
        idx = np.random.randint(0, self.size, n)
        t = lambda x: torch.from_numpy(x[idx]).to(device)
        return t(self.obs).float(), t(self.a), t(self.r), t(self.nobs).float(), t(self.d)
