"""Model-free DQN baseline on the same stacked 64x64 frames.

    python -m training.train_controller --mode dqn [--seed 0]

(The world-model controller is random-shooting MPC and needs no training; see
models/controller.py and evaluation/evaluate.py --exp control.)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from env.environment import make_env, run_episodes
from models.controller import QNetwork, ReplayBuffer
from utils import (CSVLogger, base_parser, ckpt_path, get_device, load_config, results_dir,
                   set_seed, write_json)


def greedy_fn(q, device):
    @torch.no_grad()
    def act(obs):
        return int(q(torch.from_numpy(obs).unsqueeze(0).to(device).float()).argmax(1))
    return act


def train_dqn(cfg: dict, seed: int, device) -> str:
    d = cfg["dqn"]
    set_seed(seed)
    rng = np.random.RandomState(seed)
    env, eval_env = make_env(cfg, seed), make_env(cfg, seed + 999)
    A, C, S = env.action_dim, cfg["env"]["frame_stack"], cfg["env"]["image_size"]
    q = QNetwork(C, A).to(device)
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=d["lr"])
    buf = ReplayBuffer(d["replay_size"], (C, S, S))
    tag = f"dqn_s{seed}"
    log = CSVLogger(results_dir(cfg, "logs") / f"{tag}.csv")

    obs = env.reset(seed=seed)
    ep_ret, ep_rets, loss_val = 0.0, [], float("nan")
    for step in range(1, d["total_steps"] + 1):
        eps = max(d["eps_end"], 1.0 - (1.0 - d["eps_end"]) * step / d["eps_decay_steps"])
        if rng.rand() < eps:
            a = int(rng.randint(A))
        else:
            a = greedy_fn(q, device)(obs)
        nobs, r, term, trunc = env.step(a)
        buf.add(obs, a, r, nobs, float(term))
        obs, ep_ret = nobs, ep_ret + r
        if term or trunc:
            ep_rets.append(ep_ret)
            obs, ep_ret = env.reset(), 0.0

        if step > d["learning_starts"] and step % d["train_every"] == 0:
            o, ac, rw, no, dn = buf.sample(d["batch_size"], device)
            with torch.no_grad():
                y = rw + d["gamma"] * (1 - dn) * target(no).max(1).values
            loss = F.smooth_l1_loss(q(o).gather(1, ac[:, None]).squeeze(1), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            loss_val = loss.item()
        if step % d["target_update"] == 0:
            target.load_state_dict(q.state_dict())
        if step % d["eval_every"] == 0:
            ev, _ = run_episodes(eval_env, greedy_fn(q, device), d["eval_episodes"], seed0=50_000)
            tr = float(np.mean(ep_rets[-10:])) if ep_rets else float("nan")
            log.log(step=step, epsilon=eps, train_return_last10=tr, eval_return=float(np.mean(ev)), loss=loss_val)
            print(f"[dqn s{seed}] step {step} eps {eps:.2f} train10 {tr:.1f} eval {np.mean(ev):.1f}")

    torch.save(dict(state=q.state_dict(), init=dict(in_ch=C, action_dim=A), seed=seed),
               ckpt_path(cfg, tag, "dqn.pt"))
    ev, ln = run_episodes(eval_env, greedy_fn(q, device), d["final_eval_episodes"], seed0=60_000)
    write_json(results_dir(cfg) / f"dqn_eval_s{seed}.json",
               dict(seed=seed, returns=ev, lengths=ln, mean=float(np.mean(ev)), std=float(np.std(ev))))
    print(f"[dqn s{seed}] final eval mean return {np.mean(ev):.1f} +- {np.std(ev):.1f}")
    return tag


def main():
    p = base_parser("Train model-free baseline")
    p.add_argument("--mode", default="dqn", choices=["dqn"])
    args = p.parse_args()
    cfg = load_config(args.config)
    train_dqn(cfg, cfg["seed"] if args.seed is None else args.seed, get_device(cfg))


if __name__ == "__main__":
    main()
