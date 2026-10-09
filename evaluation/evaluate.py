"""Experiments 1-5.

    python -m evaluation.evaluate --exp wm       # Exp 1-3 on one checkpoint (--tag)
    python -m evaluation.evaluate --exp probe    # linear probe z->state, imagined-latent probe, traversal
    python -m evaluation.evaluate --exp sweep    # Exp 4: data-size sweep (trains missing models)
    python -m evaluation.evaluate --exp control  # Exp 5: MPC vs horizon, random, DQN
    python -m evaluation.evaluate --exp link     # MPC return for every sweep model (accuracy -> control)
    python -m evaluation.evaluate --exp all      # wm + sweep + control + link, then plots
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data.collect import load_data, split_episodes
from env.environment import make_env, run_episodes
from models.controller import MPCController, QNetwork
from models.io import load_dynamics, load_vae
from models.vae import to_float
from training.train_dynamics import build_windows, dynamics_losses, encode_episodes, masked_mean, train_dynamics
from training.train_vae import train_vae
from utils import (ROOT, base_parser, ckpt_path, data_path, get_device, load_config, make_tag, resolve_run,
                   results_dir, set_seed, write_csv, write_json)


# ================================================================== Exp 1 & 2
@torch.no_grad()
def recon_metrics(vae, data, device, batch=1024) -> dict:
    obs = torch.from_numpy(data["obs"])
    se = kl = se_s = 0.0
    n = len(obs)
    for i in range(0, n, batch):
        x = to_float(obs[i:i + batch].to(device))
        mu, logvar = vae.encode(x)
        se += ((vae.decode(mu) - x) ** 2).mean(dim=(1, 2, 3)).sum().item()
        se_s += ((vae.decode(vae.reparameterize(mu, logvar)) - x) ** 2).mean(dim=(1, 2, 3)).sum().item()
        kl += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).sum().item()
    return dict(recon_mse=se / n, recon_mse_sampled=se_s / n, kl=kl / n)


@torch.no_grad()
def onestep_metrics(dyn, eps, cfg, device, zvar: float) -> dict:
    z, a, zn, r, d, m = build_windows(eps, cfg["dynamics"]["seq_len"], device)
    out, _ = dyn(z, a)
    mse = masked_mean(((zn - out["mu"]) ** 2).mean(-1), m).item()
    persist = masked_mean(((zn - z) ** 2).mean(-1), m).item()
    nll = masked_mean((0.5 * ((zn - out["mu"]) / out["logstd"].exp()) ** 2 + out["logstd"]).mean(-1), m).item()
    return dict(onestep_mse=mse, onestep_nmse=mse / zvar, persist_onestep_mse=persist,
                persist_onestep_nmse=persist / zvar, onestep_nll=nll)


# ================================================================== Exp 3
@torch.no_grad()
def multistep_metrics(vae, dyn, eps, cfg, device, zvar: float, sample: bool = True, seed: int = 0):
    """Per-horizon error between imagined and real (encoded) futures. Rows for h=1..Hmax.

    Each rollout starts from the real z_s with h=0 and the *real* action sequence, so the
    only difference to reality is the model's predictions.
    """
    ec = cfg["eval"]
    Hmax = max(ec["horizons"])
    starts = [(i, s) for i, e in enumerate(eps) for s in range(len(e["actions"]) - Hmax + 1)]
    if not starts:
        raise RuntimeError(f"No test episode is >= {Hmax} steps long; collect longer episodes.")
    rng = np.random.RandomState(seed)
    if len(starts) > ec["max_starts"]:
        starts = [starts[j] for j in rng.choice(len(starts), ec["max_starts"], replace=False)]
    torch.manual_seed(seed)

    keys = ["lat", "pix", "plat", "ppix", "floor", "std"]
    s1 = {k: np.zeros(Hmax) for k in keys}   # sum over starts
    s2 = {k: np.zeros(Hmax) for k in keys}   # sum of squares over starts
    for c in range(0, len(starts), ec["chunk"]):
        chunk = starts[c:c + ec["chunk"]]
        z0 = torch.tensor(np.stack([eps[i]["z"][s] for i, s in chunk]), device=device)
        acts = torch.tensor(np.stack([eps[i]["actions"][s:s + Hmax] for i, s in chunk]), device=device)
        zt = torch.tensor(np.stack([eps[i]["z"][s + 1:s + Hmax + 1] for i, s in chunk]), device=device)
        ft = torch.tensor(np.stack([eps[i]["frames"][s + 1:s + Hmax + 1] for i, s in chunk]), device=device)
        ft = to_float(ft)  # (B,H,C,h,w)
        B = len(chunk)
        pix = lambda zz: ((vae.decode(zz.reshape(-1, zz.shape[-1])).view_as(ft) - ft) ** 2).mean(dim=(2, 3, 4))
        lat = torch.zeros(B, Hmax, device=device)
        pxl = torch.zeros(B, Hmax, device=device)
        sdv = torch.zeros(B, Hmax, device=device)
        for _ in range(ec["n_samples"] if sample else 1):
            zp, _, _, _, sd = dyn.imagine(z0, acts, None, sample, return_std=True)
            lat += ((zp - zt) ** 2).mean(-1)
            pxl += pix(zp)
            sdv += sd.mean(-1)
        k = ec["n_samples"] if sample else 1
        lat, pxl, sdv = lat / k, pxl / k, sdv / k
        plat = ((z0[:, None] - zt) ** 2).mean(-1)
        ppix = pix(z0[:, None].expand_as(zt))
        floor = pix(zt)
        for name, v in zip(keys, [lat, pxl, plat, ppix, floor, sdv]):
            v = v.cpu().numpy().astype(np.float64)
            s1[name] += v.sum(0); s2[name] += (v ** 2).sum(0)
    n = len(starts)
    rows = []
    for h in range(Hmax):
        row = dict(horizon=h + 1, mode="sampled" if sample else "mean", n_starts=n)
        for name, long in zip(keys, ["latent_mse", "pixel_mse", "persist_latent_mse",
                                     "persist_pixel_mse", "recon_floor_pixel_mse", "pred_std"]):
            mean = s1[name][h] / n
            std = np.sqrt(max(s2[name][h] / n - mean ** 2, 0.0))
            row[long], row[long + "_sem"] = mean, std / np.sqrt(n)
        row["latent_nmse"] = row["latent_mse"] / zvar
        row["latent_nmse_sem"] = row["latent_mse_sem"] / zvar
        row["persist_latent_nmse"] = row["persist_latent_mse"] / zvar
        row["rmse_norm"] = float(np.sqrt(row["latent_nmse"]))
        row["pred_std_norm"] = row["pred_std"] / float(np.sqrt(zvar))
        rows.append(row)
    return rows


@torch.no_grad()
def filmstrip(vae, dyn, eps, cfg, device, seed: int = 0):
    """Real frames, VAE reconstruction of real z, and imagined frames at chosen horizons."""
    Hmax = max(cfg["eval"]["horizons"])
    for e in eps:
        if len(e["actions"]) >= Hmax:
            break
    else:
        return None
    torch.manual_seed(seed)
    z0 = torch.tensor(e["z"][:1], device=device)
    acts = torch.tensor(e["actions"][None, :Hmax], device=device)
    zp, _, _, _ = dyn.imagine(z0, acts, None, True)
    zt = torch.tensor(e["z"][1:Hmax + 1], device=device)
    hs = [h for h in cfg["eval"]["horizons"]]
    sel = [h - 1 for h in hs]
    last = lambda x: x[:, -1]  # newest frame of the stack
    return dict(horizons=hs,
                real=last(torch.from_numpy(e["frames"][1:Hmax + 1]))[sel].numpy() / 255.0,
                recon=last(vae.decode(zt[sel])).cpu().numpy(),
                imagined=last(vae.decode(zp[0, sel])).cpu().numpy())


def wm_metrics(cfg, vae, dyn, device, data=None, eps=None):
    """All world-model metrics for one (vae, dynamics) pair on the held-out test set."""
    if data is None:
        data = load_data(data_path(cfg, "test"))
    if eps is None:
        eps = split_episodes(data)
        encode_episodes(vae, eps, device)
    zvar = float(np.concatenate([e["z"] for e in eps]).var(0).mean())
    return dict(
        data=data, eps=eps, zvar=zvar,
        recon=recon_metrics(vae, data, device),
        onestep=onestep_metrics(dyn, eps, cfg, device, zvar),
        multistep=multistep_metrics(vae, dyn, eps, cfg, device, zvar, True),
        multistep_mean=multistep_metrics(vae, dyn, eps, cfg, device, zvar, False),
    )


def exp_wm(cfg, tag, device):
    vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
    m = wm_metrics(cfg, vae, dyn, device)
    out = results_dir(cfg)
    write_json(out / f"exp12_vae_onestep_{tag}.json", dict(tag=tag, latent_var=m["zvar"], **m["recon"], **m["onestep"]))
    write_csv(out / f"exp3_multistep_{tag}.csv", m["multistep"] + m["multistep_mean"])
    fs = filmstrip(vae, dyn, m["eps"], cfg, device)
    if fs is not None:
        np.savez(out / f"filmstrip_{tag}.npz", **{k: np.array(v) for k, v in fs.items()})
    print(f"[wm {tag}] recon_mse {m['recon']['recon_mse']:.5f} | one-step nmse {m['onestep']['onestep_nmse']:.4f} "
          f"(copy-z baseline {m['onestep']['persist_onestep_nmse']:.4f})")
    for r in m["multistep"]:
        if r["horizon"] in cfg["eval"]["horizons"]:
            print(f"    H={r['horizon']:2d} latent nmse {r['latent_nmse']:.4f} +- {r['latent_nmse_sem']:.4f} "
                  f"| pixel mse {r['pixel_mse']:.5f}")
    return m


# ================================================================== Latent analysis (probe + traversal)
STATE_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]


def _r2(pred, y):
    return 1.0 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


@torch.no_grad()
def exp_probe(cfg, tag, device):
    """What does z encode, and does imagination keep it?

    A linear probe (least squares + bias) maps z -> true CartPole state. It is fit on real
    training latents only, then scored (R^2) on (a) real test latents and (b) latents imagined
    h steps ahead against the true state at t+h. Also saves a latent-traversal grid.
    """
    if cfg["env"]["type"] != "cartpole":
        print("[probe] needs true state; CartPole only")
        return None
    vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
    tr = split_episodes(load_data(data_path(cfg, "train")))
    te = split_episodes(load_data(data_path(cfg, "test")))
    encode_episodes(vae, tr, device)
    encode_episodes(vae, te, device)
    X = np.concatenate([e["z"] for e in tr]); Y = np.concatenate([e["states"] for e in tr])
    aug = lambda z: np.concatenate([z, np.ones((len(z), 1), z.dtype)], 1)
    W, *_ = np.linalg.lstsq(aug(X), Y, rcond=None)
    rows = []
    zr = np.concatenate([e["z"] for e in te]); yr = np.concatenate([e["states"] for e in te])
    rows.append(dict(horizon=0, source="real", **dict(zip(STATE_NAMES, _r2(aug(zr) @ W, yr)))))

    Hmax = max(cfg["eval"]["horizons"])
    starts = [(i, s) for i, e in enumerate(te) for s in range(len(e["actions"]) - Hmax + 1)]
    rng = np.random.RandomState(0)
    if len(starts) > cfg["eval"]["max_starts"]:
        starts = [starts[j] for j in rng.choice(len(starts), cfg["eval"]["max_starts"], replace=False)]
    torch.manual_seed(0)
    z0 = torch.tensor(np.stack([te[i]["z"][s] for i, s in starts]), device=device)
    acts = torch.tensor(np.stack([te[i]["actions"][s:s + Hmax] for i, s in starts]), device=device)
    ys = np.stack([te[i]["states"][s + 1:s + Hmax + 1] for i, s in starts])
    zp = dyn.imagine(z0, acts, None, True)[0].cpu().numpy()
    for h in cfg["eval"]["horizons"]:
        pred = aug(zp[:, h - 1]) @ W
        rows.append(dict(horizon=h, source="imagined", **dict(zip(STATE_NAMES, _r2(pred, ys[:, h - 1])))))
    write_csv(results_dir(cfg) / f"probe_{tag}.csv", rows)

    # latent traversal: vary the k highest-variance dims of z around a real frame
    k, steps = 6, np.linspace(-3, 3, 7)
    dims = np.argsort(-zr.var(0))[:k]
    base = torch.tensor(te[0]["z"][:1], device=device)
    grid = []
    for d in dims:
        zz = base.repeat(len(steps), 1)
        zz[:, d] = base[0, d] + torch.tensor(steps * zr[:, d].std(), dtype=zz.dtype, device=device)
        grid.append(vae.decode(zz)[:, -1].cpu().numpy())
    np.savez(results_dir(cfg) / f"traversal_{tag}.npz", grid=np.array(grid), dims=dims, steps=steps)
    for r in rows:
        print(f"[probe] {r['source']:8s} H={r['horizon']:2d} R2 " +
              " ".join(f"{n}={r[n]:.2f}" for n in STATE_NAMES))
    return rows


# ================================================================== Exp 4
def exp_sweep(cfg, device, force=False):
    rows = []
    for n in cfg["data"]["sizes"]:
        for seed in cfg["seeds"]:
            tag = make_tag(n, seed)
            if force or not ckpt_path(cfg, tag, "vae.pt").exists():
                train_vae(cfg, n, seed, tag, device)
            if force or not ckpt_path(cfg, tag, "dynamics.pt").exists():
                train_dynamics(cfg, n, seed, tag, device)
            vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
            m = wm_metrics(cfg, vae, dyn, device)
            row = dict(n=n, seed=seed, tag=tag, **m["recon"], **m["onestep"])
            for r in m["multistep"]:
                if r["horizon"] in cfg["eval"]["horizons"]:
                    row[f"nmse_h{r['horizon']}"] = r["latent_nmse"]
                    row[f"pixel_mse_h{r['horizon']}"] = r["pixel_mse"]
            rows.append(row)
            print(f"[sweep] n={n} seed={seed} onestep_nmse={row['onestep_nmse']:.4f} "
                  f"nmse_h10={row.get('nmse_h10', float('nan')):.4f}")
            write_csv(results_dir(cfg) / "exp4_datasize.csv", rows)  # incremental save
    return rows


# ================================================================== Exp 5 / link
def _random_fn(A, seed):
    rng = np.random.RandomState(seed)
    return lambda obs: int(rng.randint(A))


def mpc_returns(cfg, vae, dyn, device, horizon, episodes, seed):
    env = make_env(cfg, seed)
    mc = cfg["mpc"]
    ctrl = MPCController(vae, dyn, env.action_dim, horizon, mc["candidates"], mc["context"], device, mc["sample"])
    set_seed(seed)
    return run_episodes(env, ctrl.act, episodes, seed0=seed * 1000, on_reset=ctrl.reset)


def exp_control(cfg, tag, device, episodes=None):
    episodes = episodes or cfg["control"]["episodes"]
    seed = cfg["seed"]
    vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
    env = make_env(cfg, seed)
    rows = []

    def add(method, horizon, rets, lens):
        rows.extend(dict(method=method, horizon=horizon, episode=i, ret=r, length=l)
                    for i, (r, l) in enumerate(zip(rets, lens)))
        print(f"[control] {method:7s} H={horizon:2d} return {np.mean(rets):7.2f} +- {np.std(rets):.2f}")

    add("random", 0, *run_episodes(env, _random_fn(env.action_dim, seed), episodes, seed0=seed * 1000))
    for H in cfg["control"]["horizons"]:
        add("mpc", H, *mpc_returns(cfg, vae, dyn, device, H, episodes, seed))
    for dq in sorted((ROOT / "checkpoints" / cfg["run_name"]).glob("dqn_s*/dqn.pt")):
        ck = torch.load(dq, map_location=device)
        q = QNetwork(**ck["init"]).to(device)
        q.load_state_dict(ck["state"])
        q.eval()
        from training.train_controller import greedy_fn
        add(dq.parent.name, 0, *run_episodes(env, greedy_fn(q, device), episodes, seed0=seed * 1000))
    write_csv(results_dir(cfg) / "exp5_control.csv", rows)
    return rows


def exp_link(cfg, device):
    """Accuracy -> control: MPC return for every sweep model, next to its prediction error."""
    sweep = pd.read_csv(results_dir(cfg) / "exp4_datasize.csv")
    H = cfg["mpc"]["horizon"]
    out = []
    for _, r in sweep.iterrows():
        vae, dyn = load_vae(cfg, r["tag"], device), load_dynamics(cfg, r["tag"], device)
        rets, _ = mpc_returns(cfg, vae, dyn, device, H, cfg["control"]["link_episodes"], int(r["seed"]))
        row = r.to_dict()
        row.update(mpc_horizon=H, mpc_return_mean=float(np.mean(rets)), mpc_return_std=float(np.std(rets)))
        out.append(row)
        print(f"[link] n={int(r['n'])} seed={int(r['seed'])} return {np.mean(rets):.1f}")
        write_csv(results_dir(cfg) / "exp_link.csv", out)
    return out


# ================================================================== CLI
def main():
    p = base_parser("Run experiments")
    p.add_argument("--exp", required=True, choices=["wm", "probe", "sweep", "control", "link", "all"])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--force", action="store_true", help="retrain sweep models even if checkpoints exist")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    _, _, tag = resolve_run(cfg, args)
    if args.exp in ("wm", "all"):
        exp_wm(cfg, tag, device)
    if args.exp in ("probe", "all"):
        exp_probe(cfg, tag, device)
    if args.exp in ("sweep", "all"):
        exp_sweep(cfg, device, args.force)
    if args.exp in ("control", "all"):
        exp_control(cfg, tag, device, args.episodes)
    if args.exp in ("link", "all"):
        exp_link(cfg, device)
    if args.exp == "all":
        from visualization.plots import make_all
        make_all(cfg, tag)


if __name__ == "__main__":
    main()
