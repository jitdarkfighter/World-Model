"""Shared helpers: config loading, seeding, paths, CSV/JSON logging."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


# ------------------------------------------------------------------ config
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> dict:
    """Load configs/config.yaml, then merge an optional override file on top."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path:
        p = Path(path)
        if not p.exists():
            p = ROOT / path
        if p.resolve() != DEFAULT_CONFIG.resolve():
            with open(p) as f:
                cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=None, help="override yaml merged over configs/config.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n", type=int, default=None, help="number of training transitions")
    p.add_argument("--tag", default=None, help="checkpoint tag (default: n{N}_s{seed})")
    return p


def make_tag(n: int, seed: int) -> str:
    return f"n{n}_s{seed}"


def resolve_run(cfg: dict, args) -> tuple[int, int, str]:
    seed = cfg["seed"] if args.seed is None else args.seed
    n = cfg["data"]["train_transitions"] if args.n is None else args.n
    tag = args.tag or make_tag(n, seed)
    return n, seed, tag


# ------------------------------------------------------------------ seeding / device
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


# ------------------------------------------------------------------ paths (always relative to project root)
def _dir(*parts) -> Path:
    p = ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_path(cfg: dict, split: str) -> Path:
    return _dir("data", "cache", cfg["run_name"]) / f"{split}.npz"


def ckpt_path(cfg: dict, tag: str, name: str) -> Path:
    return _dir("checkpoints", cfg["run_name"], tag) / name


def results_dir(cfg: dict, *sub: str) -> Path:
    return _dir("results", cfg["run_name"], *sub)


def figures_dir(cfg: dict) -> Path:
    return results_dir(cfg, "figures")


# ------------------------------------------------------------------ logging
def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class CSVLogger:
    """Truncates the file on creation, then appends one row per call."""

    def __init__(self, path: Path):
        self.path = path
        self.keys: list[str] | None = None
        if path.exists():
            path.unlink()

    def log(self, **row) -> None:
        new = self.keys is None
        if new:
            self.keys = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.keys)
            if new:
                w.writeheader()
            w.writerow(row)
