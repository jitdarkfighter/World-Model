"""Collect transitions and provide dataset helpers.

    python -m data.collect [--config configs/atari.yaml]

Writes data/cache/<run_name>/{train,test}.npz.
"""
from __future__ import annotations

import numpy as np

from env.environment import make_env
from utils import base_parser, data_path, load_config, set_seed


def collect_transitions(cfg: dict, n: int, seed: int, hfrac: float | None = None) -> dict:
    env = make_env(cfg, seed)
    rng = np.random.RandomState(seed)
    dcfg = cfg["data"]
    can_heur = env.kind == "cartpole" or dcfg.get("pong_heuristic", False)
    obs_l, nobs_l, a_l, r_l, d_l, e_l = [], [], [], [], [], []
    st_l, nst_l = [], []  # true CartPole state (analysis/probing only; zeros for Atari)
    raw = (lambda: env.raw_state()) if env.kind == "cartpole" else (lambda: np.zeros(4))
    ep = 0
    while len(a_l) < n:
        use_h = can_heur and rng.rand() < (dcfg["heuristic_fraction"] if hfrac is None else hfrac)
        obs = env.reset(seed=int(rng.randint(2**31 - 1)))
        while len(a_l) < n:
            if use_h and rng.rand() >= dcfg["heuristic_epsilon"]:
                if env.kind == "atari":
                    a = env.pong_heuristic_action()
                else:
                    s = env.raw_state()
                    a = int(s[2] + s[3] > 0)  # push towards the lean of the pole
            else:
                a = int(rng.randint(env.action_dim))
            st = raw()
            nobs, r, term, trunc = env.step(a)
            st_l.append(st); nst_l.append(raw())
            obs_l.append(obs); nobs_l.append(nobs); a_l.append(a)
            r_l.append(r); d_l.append(term); e_l.append(ep)
            obs = nobs
            if term or trunc:
                break
        ep += 1
    env.close()
    return dict(
        obs=np.stack(obs_l), next_obs=np.stack(nobs_l),
        actions=np.array(a_l, dtype=np.int64), rewards=np.array(r_l, dtype=np.float32),
        dones=np.array(d_l, dtype=np.float32), ep_id=np.array(e_l, dtype=np.int64),
        states=np.stack(st_l).astype(np.float32), next_states=np.stack(nst_l).astype(np.float32),
        action_dim=np.array(env.action_dim),
    )


def save_data(path, data: dict) -> None:
    np.savez_compressed(path, **data)


def load_data(path, n: int | None = None) -> dict:
    """Load a dataset; if n is given keep only the first n transitions."""
    z = np.load(path)
    data = {k: z[k] for k in z.files}
    if n is not None:
        for k in data:
            if k != "action_dim":
                data[k] = data[k][:n]
    return data


def split_episodes(data: dict) -> list[dict]:
    """List of episodes: frames (T+1,C,H,W), states (T+1,4), actions (T), rewards (T), dones (T)."""
    ids = data["ep_id"]
    bounds = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ids)]])
    eps = []
    for s, e in zip(starts, ends):
        eps.append(dict(
            frames=np.concatenate([data["obs"][s:e], data["next_obs"][e - 1:e]]),
            actions=data["actions"][s:e], rewards=data["rewards"][s:e], dones=data["dones"][s:e],
            states=np.concatenate([data["states"][s:e], data["next_states"][e - 1:e]]),
        ))
    return eps


def main():
    args = base_parser("Collect train/test transitions").parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed)
    dc = cfg["data"]
    for split, n, s, hf in [("train", max(dc["sizes"] + [dc["train_transitions"]]), seed, None),
                            ("test", dc["test_transitions"], seed + 10_000,
                             dc.get("test_heuristic_fraction"))]:   # test needs long episodes even if train is random-only
        data = collect_transitions(cfg, n, s, hf)
        save_data(data_path(cfg, split), data)
        n_ep = int(data["ep_id"].max()) + 1
        print(f"[{split}] {len(data['actions'])} transitions, {n_ep} episodes, "
              f"mean episode length {len(data['actions']) / n_ep:.1f}, obs {data['obs'].shape}")


if __name__ == "__main__":
    main()
