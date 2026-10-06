"""Train the probabilistic GRU dynamics on frozen-VAE latents.

    python -m training.train_dynamics [--n 20000] [--seed 0] [--tag TAG]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data.collect import load_data, split_episodes
from models.dynamics import LatentDynamics
from models.io import load_vae
from utils import (CSVLogger, base_parser, ckpt_path, data_path, get_device, load_config,
                   resolve_run, results_dir, set_seed)


@torch.no_grad()
def encode_episodes(vae, episodes: list[dict], device, batch: int = 1024) -> None:
    """Adds e['z'] (T+1, D) = posterior means of every frame, in place."""
    frames = torch.from_numpy(np.concatenate([e["frames"] for e in episodes]))
    zs = torch.cat([vae.encode_mean(frames[i:i + batch].to(device)).cpu()
                    for i in range(0, len(frames), batch)])
    i = 0
    for e in episodes:
        n = len(e["frames"])
        e["z"] = zs[i:i + n].numpy()
        i += n


def build_windows(episodes: list[dict], L: int, device):
    """All length-L windows (stride 1) with padding + mask. Returns tensors on `device`."""
    Z, ZN, A, R, D, M = [], [], [], [], [], []
    for e in episodes:
        z, T = e["z"], len(e["actions"])
        zd = z.shape[1]
        pad = lambda x, shape, dt: np.concatenate([x, np.zeros(shape, dt)])
        zi = pad(z[:-1], (L, zd), np.float32)
        zn = pad(z[1:], (L, zd), np.float32)
        a = pad(e["actions"], (L,), np.int64)
        r = pad(e["rewards"], (L,), np.float32)
        d = pad(e["dones"], (L,), np.float32)
        m = pad(np.ones(T, np.float32), (L,), np.float32)
        for s in range(T):
            sl = slice(s, s + L)
            Z.append(zi[sl]); ZN.append(zn[sl]); A.append(a[sl])
            R.append(r[sl]); D.append(d[sl]); M.append(m[sl])
    t = lambda x, dt: torch.from_numpy(np.stack(x)).to(dtype=dt, device=device)
    return (t(Z, torch.float32), t(A, torch.long), t(ZN, torch.float32),
            t(R, torch.float32), t(D, torch.float32), t(M, torch.float32))


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1.0)


def dynamics_losses(model, batch, dcfg):
    z, a, zn, r, d, m = batch
    out, _ = model(z, a)
    nll = (0.5 * ((zn - out["mu"]) / out["logstd"].exp()) ** 2 + out["logstd"]).mean(-1)
    mse = ((zn - out["mu"]) ** 2).mean(-1)
    rew = (out["reward"] - r) ** 2
    pw = torch.tensor(dcfg["done_pos_weight"], device=z.device)
    done = F.binary_cross_entropy_with_logits(out["done_logit"], d, pos_weight=pw, reduction="none")
    nll_m, mse_m = masked_mean(nll, m), masked_mean(mse, m)
    loss = nll_m + dcfg["reward_weight"] * masked_mean(rew, m) + dcfg["done_weight"] * masked_mean(done, m)
    return loss, nll_m, mse_m


def train_dynamics(cfg: dict, n: int, seed: int, tag: str, device) -> None:
    dc = cfg["dynamics"]
    set_seed(seed)
    data = load_data(data_path(cfg, "train"), n)
    action_dim = int(data["action_dim"])
    vae = load_vae(cfg, tag, device)
    eps = split_episodes(data)
    encode_episodes(vae, eps, device)
    nval = max(1, int(len(eps) * dc["val_fraction"])) if len(eps) > 1 else 0
    tr_eps, va_eps = (eps[:-nval], eps[-nval:]) if nval else (eps, eps)
    train = build_windows(tr_eps, dc["seq_len"], device)
    val = build_windows(va_eps, dc["seq_len"], device)

    model = LatentDynamics(vae.latent_dim, action_dim, dc["hidden"], dc["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=dc["lr"])
    log = CSVLogger(results_dir(cfg, "logs") / f"dynamics_{tag}.csv")
    W = train[0].shape[0]

    for epoch in range(1, dc["epochs"] + 1):
        model.train()
        idx = torch.randperm(W, device=device)
        tot, cnt = 0.0, 0
        for i in range(0, W, dc["batch_size"]):
            b = idx[i:i + dc["batch_size"]]
            loss, _, _ = dynamics_losses(model, [x[b] for x in train], dc)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), dc["grad_clip"])
            opt.step()
            tot += loss.item() * len(b); cnt += len(b)
        model.eval()
        with torch.no_grad():
            vloss, vnll, vmse = dynamics_losses(model, val, dc)
        log.log(epoch=epoch, train_loss=tot / cnt, val_loss=vloss.item(),
                val_nll=vnll.item(), val_onestep_mse=vmse.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"[dyn {tag}] epoch {epoch:3d} train {tot / cnt:.4f} val_nll {vnll.item():.4f} "
                  f"val_mse {vmse.item():.5f}")

    torch.save(dict(state=model.state_dict(),
                    init=dict(z_dim=vae.latent_dim, action_dim=action_dim, hidden=dc["hidden"],
                              layers=dc["layers"]),
                    tag=tag, n=n, seed=seed), ckpt_path(cfg, tag, "dynamics.pt"))


def main():
    args = base_parser("Train latent dynamics").parse_args()
    cfg = load_config(args.config)
    n, seed, tag = resolve_run(cfg, args)
    train_dynamics(cfg, n, seed, tag, get_device(cfg))


if __name__ == "__main__":
    main()
