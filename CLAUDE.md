# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Trains a Unitree G1 humanoid (12-DoF legs) to walk in MuJoCo using PPO (via `rsl_rl`), entirely on CPU —
no GPU or Isaac Gym. Physics, PD gains, joint order, and observation layout are matched to Unitree's own
`unitree_rl_gym` pipeline so policies are interchangeable with theirs.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym reference/unitree_rl_gym      # robot model + reference
git clone --depth 1 https://github.com/Lavansurendra/berkeley_humanoid reference/berkeley_humanoid  # pipeline reference
source .venv/bin/activate
```

`reference/` and `.venv/` are gitignored — they're external clones, not part of this repo's source.
The G1 MJCF model (`reference/unitree_rl_gym/resources/robots/g1_description/scene.xml`) is required
at runtime (`g1_walk/config.py` points `G1_XML` at it); nothing will run without that clone present.

## Commands

```bash
python train.py --iters 3000 --envs 1024 --name run1_full        # fresh run, full reward set
python train.py --reward minimal --iters 300 --name run2_minimal # minimal reward set, for comparison
python train.py --dr --resume logs/g1/run1_full/model_2999.pt --iters 1000 --name run1_dr  # add domain randomization, resume from checkpoint

python train_ppo.py --iters 600 --envs 1024 --name custom_full   # same, but with our own PPO (g1_walk/ppo.py) instead of rsl_rl

tensorboard --logdir logs/g1                                     # live training curves
python plot_curves.py logs/g1/run1_full                          # -> results/curves.png (also accepts multiple run dirs to overlay)

python play.py logs/g1/run1_full/model_2999.pt                   # deterministic rollout -> walk.mp4
python play.py logs/g1/run1_full/model_2999.pt --vx 0.8 --seconds 15
python play.py --unitree                                         # Unitree's pretrained policy, for comparison/validation
./view.sh logs/g1/run1_full/model_2999.pt --seconds 60            # live MuJoCo viewer (wraps mjpython; needed on macOS)

python evaluate.py logs/g1/run1_dr/model_3998.pt --dr            # robustness: % of N robots that fall in 20s, with/without randomization
python gait_stats.py logs/g1/run1_full/model_600.pt logs/g1/run2_minimal/model_599.pt  # gait-quality comparison across checkpoints (clearance, slip, jitter, ...)
```

No test suite, linter, or build step exists in this repo — validation is done empirically via
`evaluate.py` (fall rate), `gait_stats.py` (gait quality numbers), and visual inspection of rendered
videos/the live viewer.

`imageio[ffmpeg]` must be installed for MP4 output from `play.py`. If MP4 writing fails in imageio's
PyAV plugin with `expected bytes, NoneType found`, run `uv pip uninstall av` so imageio falls back to
its FFmpeg backend.

## Architecture

Three layers, each only aware of the one below it:

```
rsl_rl OnPolicyRunner (train.py)      — PPO: collects 24 steps × N envs, computes GAE, 5 epochs of clipped updates
          ↕ obs / reward / done, actions
g1_walk/env.py (G1Env, an rsl_rl VecEnv) — task logic: observations, 16 reward terms, fall/timeout detection, resets, command sampling, domain-rand params
          ↕ joint/base/foot state, PD targets
g1_walk/physics.py (LocalPhysics / ParallelPhysics) — MuJoCo only: steps many robot copies, reports raw state; no RL concepts at all
```

- `g1_walk/config.py` — single source of truth for every tunable: timing, PD gains, observation scales,
  command ranges, the two reward sets (`REWARD_SETS["minimal"]` / `["full"]`), DR ranges, PPO
  hyperparameters (`train_cfg`). Read this first when changing any numeric behavior.
- `g1_walk/physics.py` — `load_model()` loads the G1 MJCF and converts its torque motors into PD position
  actuators (`tau = kp*(target-q) - kd*dq`, computed by MuJoCo itself). `LocalPhysics` holds one
  `MjModel`/`MjData` pair per robot in a single process (used inside each worker, and directly by
  `play.py` with `visual=True` for rendering). `ParallelPhysics` fans a batch of envs out across
  `num_workers` CPU processes (`multiprocessing`, spawn context) and gathers state back; `env.py` talks
  to both through the same `step`/`reset`/`push` API, so it doesn't know which one it has.
- `g1_walk/env.py` — `G1Env` implements the `rsl_rl.env.VecEnv` interface. All arrays here are batched
  numpy over `num_envs` robots; conversions to/from torch happen only at the `step`/`get_observations`
  boundary. Key details: observations are a 47-value vector matching Unitree's `deploy_mujoco.py` layout
  exactly (ang vel, gravity/tilt, commanded vx/vy/yaw, joint pos/vel relative to default, previous
  action, sin/cos gait clock) — the **critic** additionally sees true base linear velocity as a
  privileged observation. Actions are 12 PD target offsets scaled by `action_scale`. Episodes reset on
  falling (height/tilt thresholds) or 20s timeout; timeouts are flagged in `extras["time_outs"]` so PPO
  bootstraps rather than penalizing them as failures.
- `train.py` / `play.py` / `evaluate.py` / `gait_stats.py` are thin CLI entry points over `G1Env` +
  `rsl_rl`'s `OnPolicyRunner`. `play.py:load_policy()` is the shared loader for a runner checkpoint
  (`model_*.pt`), an exported JIT policy (`policy.pt`), or Unitree's pretrained policy (`--unitree`) —
  `evaluate.py` and `gait_stats.py` both import it rather than reimplementing checkpoint loading.

## Two training paths

`train.py` uses rsl_rl's `OnPolicyRunner`; `train_ppo.py` uses `g1_walk/ppo.py`, a from-scratch PPO that
mirrors rsl_rl 5.5.1's semantics (separate actor/critic running normalizers updated only *after* the
update pass, scalar learned std, timeout bootstrap `r += γ·V(s_t)`, global advantage normalization,
clipped value loss, per-minibatch adaptive-KL learning rate). Both read the same `train_cfg()` and log the
same TensorBoard tags, so `plot_curves.py` can overlay them. Checkpoints are **not** interchangeable:
`train_ppo.py --resume` needs a checkpoint from `train_ppo.py` (marked `custom_ppo`); `play.py:load_policy`
accepts either format.

## Design decisions worth knowing before changing things

- **Training models strip visual meshes** (`load_model(..., visual=False)`): the meshes have no mass or
  collision anyway, and removing them cut per-env memory ~100× and startup time ~50×. `play.py` always
  loads with `visual=True` since it needs to render. Don't "fix" training to use the full visual model —
  it's a correctness-irrelevant 50× regression.
- **Domain-randomized friction must be set on every geom, not just the floor.** MuJoCo takes the *max*
  friction of two touching geoms (Isaac Gym averages them), so `friction_range: (0.55, 1.125)` is already
  the Isaac-equivalent *effective* value, applied directly to all geoms including the feet.
- **Pushes set velocity, they don't add to it** (`push()` assigns `qvel[:2]`, like legged_gym) — adding
  instead compounds across repeated pushes and was one of the bugs that made an early DR run too harsh.
- **`--dr` is a second training stage**, resumed from a stage-1 checkpoint that already walks — training
  with randomization from scratch is not how this pipeline is meant to be used.
- Reward terms are each scaled by `self.dt` (policy step, 0.02s) before summing, matching legged_gym's
  convention; `only_positive_rewards` clips the per-step total at 0.
- Joint order, PD gains, and observation scales are intentionally copied from Unitree's config so that
  `python play.py --unitree` (their pretrained policy) can run inside this same env as a sanity check —
  if you change observation/action layout, that validation path breaks.
