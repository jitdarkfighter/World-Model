"""Extra experiments on a trained world model.

    python -m evaluation.extras --exp candidates   # planning budget: return vs MPC candidates (no retraining)
    python -m evaluation.extras --exp transfer     # same world model, new task: retrain only the reward head
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from data.collect import load_data, split_episodes
from env.environment import make_env, run_episodes, task_reward
from evaluation.evaluate import _random_fn, mpc_returns
from models.io import load_dynamics, load_vae
from training.train_dynamics import build_windows, encode_episodes, masked_mean
from utils import base_parser, data_path, get_device, load_config, resolve_run, results_dir, set_seed, write_csv


def exp_candidates(cfg, tag, device, seed_model):
    vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
    H, eps_n, rows = cfg["mpc"]["horizon"], cfg["control"]["episodes"], []
    for N in cfg["extras"]["candidates"]:
        c = copy.deepcopy(cfg)
        c["mpc"]["candidates"] = N
        rets, _ = mpc_returns(c, vae, dyn, device, H, eps_n, cfg["seed"])
        rows.append(dict(model_seed=seed_model, candidates=N, horizon=H, ret_mean=float(np.mean(rets)),
                         ret_std=float(np.std(rets))))
        print(f"[candidates] seed {seed_model} N={N:5d} H={H} return {np.mean(rets):.1f} +- {np.std(rets):.1f}")
    return rows


def retrain_reward_head(vae, dyn, cfg, n, task, device, epochs=20):
    """Relabel the stored transitions with the task reward and fit ONLY dyn.reward_head."""
    data = load_data(data_path(cfg, "train"), n)
    data["rewards"] = task_reward(task, data["next_states"])
    eps = split_episodes(data)
    encode_episodes(vae, eps, device)
    z, a, zn, r, d, m = build_windows(eps, cfg["dynamics"]["seq_len"], device)
    new = copy.deepcopy(dyn)
    for p in new.parameters():
        p.requires_grad_(False)
    for p in new.reward_head.parameters():
        p.requires_grad_(True)
    new.train()  # cuDNN GRU backward needs train mode (the model has no dropout)
    opt = torch.optim.Adam(new.reward_head.parameters(), lr=1e-3)
    W = z.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(W, device=device)
        for i in range(0, W, 256):
            b = idx[i:i + 256]
            out, _ = new(z[b], a[b])
            loss = masked_mean((out["reward"] - r[b]) ** 2, m[b])
            opt.zero_grad(); loss.backward(); opt.step()
    new.eval()
    print(f"[transfer] task={task} reward-head final mse {loss.item():.4f}")
    return new


def exp_transfer(cfg, tag, device, seed_model):
    vae, dyn = load_vae(cfg, tag, device), load_dynamics(cfg, tag, device)
    n, eps_n, rows = cfg["data"]["train_transitions"], cfg["control"]["episodes"], []
    for task in cfg["extras"]["tasks"]:
        c = copy.deepcopy(cfg)
        c["env"]["task"] = task
        new = retrain_reward_head(vae, dyn, c, n, task, device)
        rnd, _ = run_episodes(make_env(c, cfg["seed"]), _random_fn(2, cfg["seed"]), eps_n, seed0=cfg["seed"] * 1000)
        rows.append(dict(model_seed=seed_model, task=task, method="random", horizon=0,
                         ret_mean=float(np.mean(rnd)), ret_std=float(np.std(rnd))))
        for H in cfg["extras"]["transfer_horizons"]:
            for name, model in (("mpc_relabelled", new), ("mpc_stale_head", dyn)):
                set_seed(cfg["seed"])
                rets, _ = mpc_returns(c, vae, model, device, H, eps_n, cfg["seed"])
                rows.append(dict(model_seed=seed_model, task=task, method=name, horizon=H,
                                 ret_mean=float(np.mean(rets)), ret_std=float(np.std(rets))))
                print(f"[transfer] seed {seed_model} {task:8s} {name:15s} H={H:2d} return {np.mean(rets):.1f} (random {np.mean(rnd):.1f})")
    return rows


def main():
    p = base_parser("Extra experiments")
    p.add_argument("--exp", required=True, choices=["candidates", "transfer"])
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("extras", {})
    cfg["extras"] = {**dict(candidates=[16, 64, 256, 1024, 4096], tasks=["left", "upright"],
                            transfer_horizons=[10, 20]), **cfg["extras"]}
    device = get_device(cfg)
    n, seed, tag = resolve_run(cfg, args)
    rows = (exp_candidates if args.exp == "candidates" else exp_transfer)(cfg, tag, device, seed)
    write_csv(results_dir(cfg) / f"extras_{args.exp}_s{seed}.csv", rows)


if __name__ == "__main__":
    main()
