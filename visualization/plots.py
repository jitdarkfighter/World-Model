"""Figures (PNG + PDF). Reads whatever result files exist.

    python -m visualization.plots [--tag n20000_s0] [--config ...]
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import base_parser, figures_dir, load_config, resolve_run, results_dir

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.savefig(f"{path}.png")
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def _last(x):
    """(B,C,H,W) or (B,H,W) -> (B,H,W) newest frame."""
    return x[:, -1] if x.ndim == 4 else x


def save_recon_grid(orig: np.ndarray, recon: np.ndarray, path) -> None:
    """Top row original, bottom row reconstruction."""
    o, r = _last(orig), _last(recon)
    n = len(o)
    fig, ax = plt.subplots(2, n, figsize=(1.2 * n, 2.6))
    for i in range(n):
        for j, img in enumerate((o[i], r[i])):
            ax[j, i].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[j, i].axis("off")
    ax[0, 0].set_title("original", fontsize=7, loc="left")
    ax[1, 0].set_title("reconstruction", fontsize=7, loc="left")
    _save(fig, path)


def plot_filmstrip(cfg, tag):
    f = results_dir(cfg) / f"filmstrip_{tag}.npz"
    if not f.exists():
        return
    d = np.load(f)
    hs = d["horizons"]
    fig, ax = plt.subplots(3, len(hs), figsize=(1.6 * len(hs), 5))
    for j, h in enumerate(hs):
        for i, key in enumerate(["real", "recon", "imagined"]):
            ax[i, j].imshow(d[key][j], cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[0, j].set_title(f"H={h}", fontsize=8)
    for i, name in enumerate(["real", "VAE recon of real z", "imagined"]):
        ax[i, 0].set_ylabel(name, fontsize=7)
    _save(fig, figures_dir(cfg) / f"exp3_filmstrip_{tag}")


def plot_horizon_error(cfg, tag):
    f = results_dir(cfg) / f"exp3_multistep_{tag}.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for mode, c in [("sampled", "C0"), ("mean", "C1")]:
        d = df[df["mode"] == mode]
        ax[0].errorbar(d["horizon"], d["latent_nmse"], yerr=d["latent_nmse_sem"], color=c, marker="o",
                       ms=3, capsize=2, label=f"imagined ({mode})")
        ax[1].errorbar(d["horizon"], d["pixel_mse"], yerr=d["pixel_mse_sem"], color=c, marker="o",
                       ms=3, capsize=2, label=f"imagined ({mode})")
    d = df[df["mode"] == "sampled"]
    ax[0].plot(d["horizon"], d["persist_latent_nmse"], "k--", label=r"copy $z_t$")
    ax[1].plot(d["horizon"], d["persist_pixel_mse"], "k--", label=r"copy $z_t$")
    ax[1].plot(d["horizon"], d["recon_floor_pixel_mse"], "g:", label="VAE recon floor")
    ax[0].set_ylabel("normalised latent MSE"); ax[1].set_ylabel("pixel MSE")
    for a in ax:
        a.set_xlabel("imagination horizon H"); a.legend(fontsize=7)
    _save(fig, figures_dir(cfg) / f"exp3_horizon_error_{tag}")


def plot_datasize(cfg):
    f = results_dir(cfg) / "exp4_datasize.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    cols = [c for c in df.columns if c.startswith("nmse_h")] + ["onestep_nmse"]
    g = df.groupby("n")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for c in sorted(cols, key=lambda s: (s != "onestep_nmse", len(s), s)):
        m, s = g[c].mean(), g[c].std().fillna(0)
        label = "one-step" if c == "onestep_nmse" else "H=" + c.split("_h")[1]
        ax[0].errorbar(m.index, m.values, yerr=s.values, marker="o", ms=3, capsize=2, label=label)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("training transitions"); ax[0].set_ylabel("normalised latent MSE")
    ax[0].legend(fontsize=7)
    m, s = g["recon_mse"].mean(), g["recon_mse"].std().fillna(0)
    ax[1].errorbar(m.index, m.values, yerr=s.values, marker="o", ms=3, capsize=2, color="C3")
    ax[1].set_xscale("log"); ax[1].set_xlabel("training transitions"); ax[1].set_ylabel("VAE recon MSE (test)")
    _save(fig, figures_dir(cfg) / "exp4_datasize")


def plot_control(cfg):
    f = results_dir(cfg) / "exp5_control.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    m = df[df["method"] == "mpc"].groupby("horizon")["ret"]
    mean, sem = m.mean(), m.std() / np.sqrt(m.count())
    ax.errorbar(mean.index, mean.values, yerr=sem.values, marker="o", ms=3, capsize=2, label="world-model MPC")
    for method, ls, c in [("random", "--", "k")] + [(k, "-.", "C3") for k in df["method"].unique() if k.startswith("dqn")]:
        d = df[df["method"] == method]["ret"]
        if len(d):
            ax.axhline(d.mean(), ls=ls, color=c, label="random" if method == "random" else "DQN (model-free)")
    ax.set_xlabel("imagination horizon H"); ax.set_ylabel("mean episode return"); ax.legend(fontsize=7)
    _save(fig, figures_dir(cfg) / "exp5_control_vs_horizon")


def plot_link(cfg):
    f = results_dir(cfg) / "exp_link.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    H = int(df["mpc_horizon"].iloc[0])
    col = f"nmse_h{H}"
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    g = df.groupby("n")["mpc_return_mean"]
    ax[0].errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0).values, marker="o", ms=3, capsize=2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("training transitions"); ax[0].set_ylabel(f"MPC return (H={H})")
    sc = ax[1].scatter(df[col], df["mpc_return_mean"], c=np.log10(df["n"]), cmap="viridis", s=18)
    ax[1].set_xscale("log"); ax[1].set_xlabel(f"normalised latent MSE at H={H}"); ax[1].set_ylabel(f"MPC return (H={H})")
    fig.colorbar(sc, ax=ax[1], label=r"$\log_{10}$ transitions")
    _save(fig, figures_dir(cfg) / "exp_link_error_vs_return")


def plot_uncertainty(cfg, tag):
    f = results_dir(cfg) / f"exp3_multistep_{tag}.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d["mode"] == "sampled"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(d["horizon"], d["rmse_norm"], marker="o", ms=3, label="actual error (RMSE)")
    ax.plot(d["horizon"], d["pred_std_norm"], marker="s", ms=3, label=r"model's predicted $\sigma$")
    ax.set_xlabel("imagination horizon H"); ax.set_ylabel("normalised latent units"); ax.legend(fontsize=7)
    _save(fig, figures_dir(cfg) / f"exp3_uncertainty_{tag}")


def plot_probe(cfg, tag):
    f = results_dir(cfg) / f"probe_{tag}.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for n in names:
        ax.plot(df["horizon"], df[n], marker="o", ms=3, label=n)
    ax.set_xlabel("horizon (0 = real encoding)"); ax.set_ylabel(r"probe $R^2$ vs true state")
    ax.legend(fontsize=7)
    _save(fig, figures_dir(cfg) / f"probe_r2_{tag}")


def plot_traversal(cfg, tag):
    f = results_dir(cfg) / f"traversal_{tag}.npz"
    if not f.exists():
        return
    d = np.load(f)
    g = d["grid"]
    fig, ax = plt.subplots(g.shape[0], g.shape[1], figsize=(1.0 * g.shape[1], 1.0 * g.shape[0]))
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax[i, j].imshow(g[i, j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(f"z[{d['dims'][i]}]", fontsize=6)
    _save(fig, figures_dir(cfg) / f"latent_traversal_{tag}")


def plot_data_efficiency(cfg):
    """MPC return vs real-environment transitions used, next to DQN's learning curve."""
    link, logs = results_dir(cfg) / "exp_link.csv", sorted((results_dir(cfg) / "logs").glob("dqn_s*.csv"))
    if not link.exists() or not logs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    df = pd.read_csv(link)
    g = df.groupby("n")["mpc_return_mean"]
    ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0).values, marker="o", ms=4,
                capsize=2, color="C0", label=f"world-model MPC (H={int(df['mpc_horizon'].iloc[0])})")
    dq = pd.concat([pd.read_csv(f) for f in logs]).groupby("step")["eval_return"]
    ax.errorbar(dq.mean().index, dq.mean().values, yerr=dq.std().fillna(0).values, color="C3", marker="s",
                ms=3, capsize=2, label="DQN (eval during training)")
    c5 = results_dir(cfg) / "exp5_control.csv"
    if c5.exists():
        r = pd.read_csv(c5)
        r = r[r["method"] == "random"]["ret"]
        if len(r):
            ax.axhline(r.mean(), color="k", ls="--", label="random")
    ax.set_xscale("log"); ax.set_xlabel("real environment transitions"); ax.set_ylabel("mean episode return")
    ax.legend(fontsize=7)
    _save(fig, figures_dir(cfg) / "data_efficiency_mpc_vs_dqn")


def make_all(cfg, tag):
    plot_uncertainty(cfg, tag)
    plot_probe(cfg, tag)
    plot_traversal(cfg, tag)
    plot_data_efficiency(cfg)
    plot_horizon_error(cfg, tag)
    plot_filmstrip(cfg, tag)
    plot_datasize(cfg)
    plot_control(cfg)
    plot_link(cfg)
    print(f"figures written to {figures_dir(cfg)}")


def main():
    args = base_parser("Generate figures").parse_args()
    cfg = load_config(args.config)
    _, _, tag = resolve_run(cfg, args)
    make_all(cfg, tag)


if __name__ == "__main__":
    main()
