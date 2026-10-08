"""Robustness test: run many robots for 20 s with random commands and count falls.

  python evaluate.py logs/g1/run1_full/model_2999.pt            # nominal conditions
  python evaluate.py logs/g1/run1_full/model_2999.pt --dr       # random friction/mass/motors + pushes
"""
import argparse

import numpy as np
import torch

from g1_walk.config import EnvCfg
from g1_walk.env import G1Env
from play import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--dr", action="store_true")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    cfg = EnvCfg(num_envs=args.envs, num_workers=8, domain_rand=args.dr, obs_noise=args.dr)
    env = G1Env(cfg, seed=args.seed)
    policy = load_policy(args.checkpoint, env)
    obs = env.get_observations()

    fell = np.zeros(args.envs, bool)
    track_err, n = 0.0, 0
    with torch.no_grad():
        for _ in range(env.max_episode_length - 1):  # one full 20 s episode per robot
            obs, _, done, extras = env.step(policy(obs))
            alive = ~fell
            track_err += np.linalg.norm(env.commands[alive, :2] - env.base_lin_vel[alive, :2], axis=1).sum()
            n += alive.sum()
            fell |= done.numpy().astype(bool) & ~extras["time_outs"].numpy()
    env.close()
    print(f"{'randomized' if args.dr else 'nominal'}: {fell.mean():.1%} of {args.envs} robots fell within 20 s, "
          f"mean velocity error {track_err / n:.3f} m/s")


if __name__ == "__main__":
    main()
