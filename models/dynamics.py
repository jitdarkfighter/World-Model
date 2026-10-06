"""Probabilistic GRU latent dynamics: p(z_{t+1} | z_t, a_t, h_t) = N(mu, sigma^2).

Also predicts reward and termination (needed by the planner).
Convention: the GRU consumes (z_t, a_t); its output h predicts z_{t+1}, r_t, done_t.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamics(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden: int = 256, layers: int = 1,
                 min_logstd: float = -5.0, max_logstd: float = 2.0):
        super().__init__()
        self.z_dim, self.action_dim, self.hidden, self.layers = z_dim, action_dim, hidden, layers
        self.min_logstd, self.max_logstd = min_logstd, max_logstd
        self.inp = nn.Sequential(nn.Linear(z_dim + action_dim, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.mu_head = nn.Linear(hidden, z_dim)        # predicts a residual: mu = z_t + delta
        self.logstd_head = nn.Linear(hidden, z_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)

    def forward(self, z, a, h0=None):
        """z (B,T,D), a (B,T) long -> dict of per-step outputs, final hidden."""
        x = torch.cat([z, F.one_hot(a, self.action_dim).float()], dim=-1)
        out, h = self.gru(self.inp(x), h0)
        logstd = self.logstd_head(out).clamp(self.min_logstd, self.max_logstd)
        return dict(
            mu=z + self.mu_head(out), logstd=logstd,
            reward=self.reward_head(out).squeeze(-1),
            done_logit=self.done_head(out).squeeze(-1),
        ), h

    @torch.no_grad()
    def imagine(self, z0, actions, h0=None, sample: bool = True, return_std: bool = False):
        """Roll out entirely inside the model.

        z0 (B,D), actions (B,H) long, h0 (layers,B,hidden) or None.
        Returns z (B,H,D) = z_1..z_H, reward (B,H), done_prob (B,H), final hidden
        (+ predicted std (B,H,D) if return_std).
        """
        B, H = actions.shape
        z, h = z0, h0
        zs, rs, ds, sds = [], [], [], []
        for t in range(H):
            out, h = self.forward(z.unsqueeze(1), actions[:, t:t + 1], h)
            mu, logstd = out["mu"][:, 0], out["logstd"][:, 0]
            z = mu + torch.randn_like(mu) * logstd.exp() if sample else mu
            zs.append(z)
            sds.append(logstd.exp())
            rs.append(out["reward"][:, 0])
            ds.append(torch.sigmoid(out["done_logit"][:, 0]))
        res = (torch.stack(zs, 1), torch.stack(rs, 1), torch.stack(ds, 1), h)
        return res + (torch.stack(sds, 1),) if return_std else res
