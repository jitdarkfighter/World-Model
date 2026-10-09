"""Train the VAE.   python -m training.train_vae [--n 20000] [--seed 0] [--tag TAG]"""
from __future__ import annotations

import torch

from data.collect import load_data
from models.vae import VAE, to_float, vae_loss
from utils import (CSVLogger, base_parser, ckpt_path, data_path, figures_dir, get_device,
                   load_config, resolve_run, results_dir, set_seed)


def train_vae(cfg: dict, n: int, seed: int, tag: str, device) -> None:
    v = cfg["vae"]
    set_seed(seed)
    data = load_data(data_path(cfg, "train"), n)
    obs = torch.from_numpy(data["obs"])
    nval = max(1, int(len(obs) * v["val_fraction"]))
    perm = torch.randperm(len(obs))
    val, train = obs[perm[:nval]].to(device), obs[perm[nval:]].to(device)

    vae = VAE(obs.shape[1], v["latent_dim"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=v["lr"])
    log = CSVLogger(results_dir(cfg, "logs") / f"vae_{tag}.csv")
    npix = obs.shape[1] * obs.shape[2] * obs.shape[3]

    for epoch in range(1, v["epochs"] + 1):
        vae.train()
        idx = torch.randperm(len(train), device=device)
        tot = rec_tot = kl_tot = 0.0
        for i in range(0, len(idx), v["batch_size"]):
            x = to_float(train[idx[i:i + v["batch_size"]]])
            recon, mu, logvar = vae(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, v["beta"], v.get("fg_weight", 0.0))
            opt.zero_grad(); loss.backward(); opt.step()
            b = len(x)
            tot += loss.item() * b; rec_tot += rec.item() * b; kl_tot += kl.item() * b
        vae.eval()
        with torch.no_grad():
            xv = to_float(val)
            mu, _ = vae.encode(xv)
            val_mse = ((vae.decode(mu) - xv) ** 2).mean().item()
        m = len(train)
        log.log(epoch=epoch, loss=tot / m, recon_sse=rec_tot / m, kl=kl_tot / m,
                train_mse_per_pixel=rec_tot / m / npix, val_mse_per_pixel=val_mse)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[vae {tag}] epoch {epoch:3d} loss {tot / m:.2f} kl {kl_tot / m:.2f} val_mse {val_mse:.5f}")

    torch.save(dict(state=vae.state_dict(), init=dict(in_channels=obs.shape[1], latent_dim=v["latent_dim"]),
                    tag=tag, n=n, seed=seed), ckpt_path(cfg, tag, "vae.pt"))
    try:
        from visualization.plots import save_recon_grid
        with torch.no_grad():
            x = to_float(val[:8])
            rec = vae.decode(vae.encode(x)[0])
        save_recon_grid(x.cpu().numpy(), rec.cpu().numpy(), figures_dir(cfg) / f"vae_recon_{tag}")
    except ImportError:
        pass


def main():
    args = base_parser("Train VAE").parse_args()
    cfg = load_config(args.config)
    n, seed, tag = resolve_run(cfg, args)
    train_vae(cfg, n, seed, tag, get_device(cfg))


if __name__ == "__main__":
    main()
