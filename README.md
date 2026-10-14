# Learning Generative World Models for Model-Based Reinforcement Learning

A lightweight study of a **VAE + probabilistic GRU latent world model** on visual CartPole
(optional: one Atari game). Not a Dreamer/MuZero re-implementation, and no novelty is claimed.

**Research question.** How does the quality of a learned generative world model affect
model-based RL performance, particularly as the imagination horizon increases?

**Hypotheses (to be tested, not results).**
- H1: more training data improves world-model prediction quality.
- H2: prediction error increases with imagination horizon.
- H3: there is an intermediate optimal horizon for model-based control (errors accumulate).
- H4: a sufficiently accurate model provides useful imagined trajectories for control.

## What we did
1. Built a CartPole-from-pixels pipeline: 64x64 grayscale frame stacks, a CNN VAE (32-d latent),
   a probabilistic GRU dynamics model (next latent, reward, done) and a random-shooting MPC
   that plans entirely in imagination.
2. Measured prediction error against imagination horizon, training-set size (1k-20k transitions,
   3 seeds) and compared against a copy-last-latent baseline.
3. Evaluated control: MPC at several horizons vs a random policy and a pixel DQN, including a
   matched-data comparison (world-model planner vs DQN with the same number of real steps).
4. Extended to Atari Pong (cropped, background-subtracted frames, foreground-weighted VAE loss,
   tracking-policy data) using the same models, only the config changes.
5. Wrote up the study as an IEEE-style paper (`../paper/`).

Headline findings (CartPole, 20k transitions; see the paper for full tables and caveats):
- Latent error grows with horizon and beats the copy baseline only up to about H=10.
- MPC with H=10 gets a mean return of about 119 vs 19 for random and about 102 for DQN (80k steps);
  H=1 is worse than random.
- With matched real-environment data the planner is **not** significantly better than DQN, so no
  sample-efficiency claim is made.
- Pong: the model beats the copy baseline but control stays weak (negative result, reported as such).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# quick sanity check (~10 min): data -> VAE -> dynamics -> evaluation -> plots
python -m data.collect && python -m training.train_vae && python -m training.train_dynamics
python -m evaluation.evaluate --exp wm && python -m visualization.plots
```
The full experiment order is under "Run order" below. Configs used for the paper:
`configs/cp2.yaml` (main CartPole), `cp_rand.yaml` (random-data study), `dqn20k.yaml` / `dqn40k.yaml`
(matched-data DQN), `pong2a.yaml` / `pong2b.yaml` (Pong). Pass one with `--config configs/<name>.yaml`.
Generated data, checkpoints and results are git-ignored.

## Architecture
```
frame stack (2x64x64, gray) -> CNN VAE -> z_t (32-d, posterior mean)
(z_t, onehot a_t) -> GRU(256) -> N(mu_{t+1}, sigma_{t+1}^2)   (+ reward, done heads)
sample z_{t+1} -> feed back -> imagined trajectory z_1..z_H
random-shooting MPC scores imagined action sequences, executes the first action
```
Baselines: random policy; DQN (CNN on the same stacked frames).

## Setup
```bash
cd Atari-WorldModel   # repo root
pip install -r requirements.txt
```
All commands run from the repo root. Paths are relative to the project; outputs go to
`data/cache/<run>/`, `checkpoints/<run>/<tag>/`, `results/<run>/` (CSV, JSON, `figures/`, `logs/`).
`<run>` is `cartpole` (default) or `pong`.

## Run order
```bash
python -m data.collect                                  # 1. train/test transitions
python -m training.train_vae                            # 2. VAE   -> results/cartpole/figures/vae_recon_*.png
python -m training.train_dynamics                       # 3. GRU dynamics
python -m evaluation.evaluate --exp wm                  # 4. Exp 1-3: recon, one-step, multi-step (H=1..20)
python -m evaluation.evaluate --exp probe               # 4b. latent analysis: linear probe, traversal
python -m evaluation.evaluate --exp sweep               # 5. Exp 4: 1k/5k/10k/20k x 3 seeds (trains models)
python -m training.train_controller --mode dqn          # 6. model-free baseline
python -m evaluation.evaluate --exp control             # 7. Exp 5: MPC vs horizon, random, DQN
python -m evaluation.evaluate --exp link                # 8. prediction error vs MPC return per sweep model
python -m visualization.plots                           # 9. figures (PNG + PDF)
```
## How to execute each script
Run everything from the repo root. Every script accepts `--config <yaml>` (merged over
`configs/config.yaml`, e.g. `configs/atari.yaml`) and, where relevant, `--seed`, `--n`, `--tag`.
Checkpoint tag defaults to `n{N}_s{seed}`, e.g. `n20000_s0`. Times are rough, unmeasured estimates.

| Script | Command | Needs | Produces | ~Time |
|---|---|---|---|---|
| `data/collect.py` | `python -m data.collect` | gymnasium, pygame | `data/cache/<run>/train.npz`, `test.npz` | 3-6 min |
| `training/train_vae.py` | `python -m training.train_vae --n 20000 --seed 0` | train.npz | `checkpoints/<run>/<tag>/vae.pt`, `results/<run>/logs/vae_<tag>.csv`, recon grid | 1-3 min |
| `training/train_dynamics.py` | `python -m training.train_dynamics --n 20000 --seed 0` | VAE with the same tag | `.../<tag>/dynamics.pt`, `logs/dynamics_<tag>.csv` | 2-4 min |
| `evaluation/evaluate.py --exp wm` | `python -m evaluation.evaluate --exp wm --tag n20000_s0` | VAE + dynamics, test.npz | `exp12_vae_onestep_<tag>.json`, `exp3_multistep_<tag>.csv`, `filmstrip_<tag>.npz` | 1-2 min |
| `evaluation/evaluate.py --exp probe` | `python -m evaluation.evaluate --exp probe --tag n20000_s0` | same (CartPole only) | `probe_<tag>.csv`, `traversal_<tag>.npz` | <1 min |
| `evaluation/evaluate.py --exp sweep` | `python -m evaluation.evaluate --exp sweep` | train.npz, test.npz | trains missing models, `exp4_datasize.csv` (add `--force` to retrain) | 20-40 min |
| `training/train_controller.py` | `python -m training.train_controller --mode dqn --seed 0` | gymnasium | `checkpoints/<run>/dqn_s0/dqn.pt`, `logs/dqn_s0.csv`, `dqn_eval_s0.json` | 15-30 min |
| `evaluation/evaluate.py --exp control` | `python -m evaluation.evaluate --exp control --tag n20000_s0 --episodes 20` | world model; DQN optional | `exp5_control.csv` (random, MPC per H, DQN) | 5-10 min |
| `evaluation/evaluate.py --exp link` | `python -m evaluation.evaluate --exp link` | `exp4_datasize.csv` | `exp_link.csv` (error + MPC return per sweep model) | 5-10 min |
| `visualization/plots.py` | `python -m visualization.plots --tag n20000_s0` | any of the above | PNG + PDF in `results/<run>/figures/` (skips figures whose inputs are missing) | seconds |

Quick single-model path (about 10 min, good for a first sanity check):
```bash
python -m data.collect && python -m training.train_vae && python -m training.train_dynamics
python -m evaluation.evaluate --exp wm && python -m visualization.plots
```
Look at `results/cartpole/figures/vae_recon_n20000_s0.png` before going further. For quicker
iteration, lower `vae.epochs` / `dynamics.epochs` or pass `--n 5000`.

Atari: append `--config configs/atari.yaml` to every command above.

`--exp all` runs 4, 4b, 5, 7, 8 and the plots (train DQN first). Useful flags: `--seed`, `--n`, `--tag`,
`--config`, `--episodes`, `--force`. Optional capacity ablation: set `dynamics.hidden` to 128/512 and
use a different `--tag`.

## Data efficiency and latent analysis (for the report)
- `data_efficiency_mpc_vs_dqn`: MPC return vs real transitions used (from `exp_link.csv`) next to the
  DQN learning curve (`logs/dqn_s*.csv`; DQN eval uses only 5 episodes per point, so it is noisy).
  Caveat: world-model data is half heuristic-driven; for a fair comparison rerun with
  `data.heuristic_fraction: 0` and say which you report.
- `probe_r2_*`: linear probe z -> true state, fit on real latents, scored on real and imagined latents
  (does imagination preserve cart/pole physics, including velocities?). `exp3_uncertainty_*`: predicted
  sigma vs actual error (does the model know it is drifting?). `latent_traversal_*`: decoded sweeps of
  the highest-variance latent dims. Probe needs the saved true state, so re-collect data after updates.

## Verifying each stage (what to expect, not what you will get)
1. `data.collect` prints episode counts and mean length; heuristic episodes should be much longer
   than the ~20-step random ones. Test set must contain episodes >= 20 steps (needed for H=20).
2. VAE: the recon grid should show the cart and pole at the right place/angle; `val_mse_per_pixel`
   in `results/<run>/logs/vae_*.csv` should fall. If the pole is blurred away, lower `vae.beta`.
3. Dynamics: `onestep_nmse` should be well below the copy-`z_t` baseline in `exp12_vae_onestep_*.json`.
4. Multi-step: `exp3_multistep_*.csv` error per horizon; compare against the copy-`z_t` baseline and
   the VAE reconstruction floor before drawing conclusions.
5. Control: compare `mpc` at each H with `random` and `dqn_s*` rows in `exp5_control.csv`.

## Notes / deliberate choices
- Latent MSE is also reported normalised by the variance of the test latents (`*_nmse`) so numbers
  are comparable across VAEs (the sweep trains a separate VAE per data size).
- Rollouts start from the real `z_s` with `h=0` and use the **real action sequence**, so error
  isolates model prediction. MPC warms the GRU on the last `mpc.context` real steps to match
  training windows.
- CartPole data: half of collection episodes use a noisy balancing heuristic (true state used for
  collection only) because pure random episodes are too short for H=20. Report this in the paper.
- Episodes are capped at `env.max_steps` (200) for all methods. Rewards are not discounted in MPC.
- Sweep seeds change initialisation/shuffling; data subsets are the first N transitions of one dataset.
- Frame stack of 2 gives velocity information to both the world model and DQN.

## Atari (only after CartPole works)
```bash
# Atari deps are already in requirements.txt
python -m data.collect --config configs/atari.yaml
python -m training.train_vae --config configs/atari.yaml
# ... same commands with --config configs/atari.yaml
```
Same models; only the env wrapper settings change. Atari MPC with random shooting over a short
horizon may be weak; if unstable, report CartPole as the final project.

## Risks / fallbacks
| Risk | Fallback |
|---|---|
| Pole thin/blurred at 64x64 | lower `vae.beta`, raise `latent_dim`, check recon grid first |
| Posterior collapse | lower `vae.beta` (default 0.1) |
| Rollout drift is huge | that is the studied effect; also report `mean` mode and copy baseline |
| H=1 MPC equals random | expected (all actions score ~equally); not a bug |
| DQN unstable from pixels | more `dqn.total_steps`, lower lr; report as observed |
| Atari too slow/unstable | drop it; CartPole is the complete project |

## References (verify details before citing)
- D. Ha and J. Schmidhuber, "World Models," arXiv:1803.10122, 2018.
- D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," ICLR 2014 (arXiv:1312.6114).
- D. Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet), ICML 2019 (arXiv:1811.04551).
- D. Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," ICLR 2020 (arXiv:1912.01603).
- D. Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2), ICLR 2021 (arXiv:2010.02193).
- D. Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), arXiv:2301.04104.
- L. Kaiser et al., "Model-Based Reinforcement Learning for Atari" (SimPLe), ICLR 2020 (arXiv:1903.00374).
- M. Janner et al., "When to Trust Your Model: Model-Based Policy Optimization," NeurIPS 2019 (arXiv:1906.08253).
- V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
- K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," 2014 (arXiv:1406.1078).

## Paper mapping
I Introduction / II Literature review (above) / III Methodology (Architecture, Notes) /
IV Results (Exp 1-5 figures in `results/<run>/figures/`) / V Conclusion. The contribution is an
experimental study of lightweight generative latent world models, not a new architecture.
