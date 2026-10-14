"""Visual environment wrapper: rendered frames -> 64x64 grayscale uint8, stacked.

One wrapper serves CartPole and Atari so the models never change.
"""
from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np


def task_reward(task: str, s) -> float:
    """CartPole reward variants computed from the true state (x, x_dot, theta, theta_dot).

    left:    1 while the cart is in the left half (x < 0), else 0 (the pole must still be balanced).
    upright: 1 - |theta| / 0.2095 (clipped at 0): reward for holding the pole near vertical.
    Works on a single state (4,) or an array (N, 4).
    """
    s = np.asarray(s)
    if task == "left":
        return (s[..., 0] < 0).astype(np.float32) if s.ndim > 1 else float(s[0] < 0)
    if task == "upright":
        v = np.clip(1.0 - np.abs(s[..., 2]) / 0.2095, 0.0, None)
        return v.astype(np.float32) if s.ndim > 1 else float(v)
    raise ValueError(task)


class VisualEnv:
    def __init__(self, cfg: dict, seed: int = 0):
        e = cfg["env"]
        self.kind = e.get("type", "cartpole")
        self.size = e["image_size"]
        self.k = e["frame_stack"]
        self.max_steps = e["max_steps"]
        self.clip = e.get("clip_reward", False)
        self.task = e.get("task", "survive")   # CartPole reward variants for the task-transfer study
        self.pong_pre = e.get("preprocess") == "pong"   # crop playfield + background subtraction
        self.last_gray = None

        if self.kind == "atari":
            import ale_py

            gym.register_envs(ale_py)
            self.env = gym.make(e["name"], frameskip=4, repeat_action_probability=0.0,
                                full_action_space=False)
        else:
            self.env = gym.make(e["name"], render_mode="rgb_array")
        self.action_dim = int(self.env.action_space.n)
        self.env.action_space.seed(seed)
        self.frames: deque = deque(maxlen=self.k)
        self.steps = 0

    # ---- frame processing
    def _process(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.last_gray = gray
        if self.kind == "atari" and self.pong_pre:
            # Pong: crop score bar/walls, subtract the flat background, boost contrast so the
            # 1-2 pixel ball survives downsampling (objects bright on black)
            field = gray[34:194].astype(np.float32)
            fg = np.clip(np.abs(field - 87.0) * 3.0, 0, 255).astype(np.uint8)
            return cv2.resize(fg, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.kind == "cartpole":
            gray = 255 - gray[150:330]  # crop to the cart/pole band, invert: objects bright on dark
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _grab(self, obs) -> np.ndarray:
        rgb = self.env.render() if self.kind == "cartpole" else obs
        return self._process(rgb)

    def _stacked(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.uint8)

    # ---- gym-like API
    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=seed)
        f = self._grab(obs)
        for _ in range(self.k):
            self.frames.append(f)
        self.steps = 0
        return self._stacked()

    def step(self, action: int):
        obs, r, term, trunc, _ = self.env.step(int(action))
        self.steps += 1
        self.frames.append(self._grab(obs))
        if self.steps >= self.max_steps:
            trunc = True
        if self.task != "survive":
            r = task_reward(self.task, self.raw_state())
        if self.clip:
            r = float(np.sign(r))
        return self._stacked(), float(r), bool(term), bool(trunc)

    def pong_heuristic_action(self) -> int:
        """Track the ball with our (right) paddle. Data collection only; uses the raw frame."""
        field = self.last_gray[34:194]
        ball = np.argwhere(field == 236)
        pad = np.argwhere((field[:, 128:] >= 146) & (field[:, 128:] <= 149))
        if len(ball) == 0 or len(pad) == 0:
            return 0
        by, py = ball[:, 0].mean(), pad[:, 0].mean()
        return 2 if by < py - 3 else 3 if by > py + 3 else 0   # 2 = up, 3 = down

    def raw_state(self) -> np.ndarray:
        """CartPole true state. Used only by the data-collection heuristic."""
        return np.array(self.env.unwrapped.state, dtype=np.float64)

    def close(self) -> None:
        self.env.close()


def make_env(cfg: dict, seed: int = 0) -> VisualEnv:
    return VisualEnv(cfg, seed)


def run_episodes(env: VisualEnv, act_fn, n: int, seed0: int = 0, on_reset=None):
    """Run n episodes; act_fn(obs)->action. Returns (returns, lengths)."""
    returns, lengths = [], []
    for i in range(n):
        obs = env.reset(seed=seed0 + i)
        if on_reset is not None:
            on_reset()
        total, steps = 0.0, 0
        while True:
            obs, r, term, trunc = env.step(act_fn(obs))
            total += r
            steps += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(steps)
    return returns, lengths
