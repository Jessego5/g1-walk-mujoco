"""Train a G1 walking policy with rsl_rl PPO on CPU.

  python train.py                                   # fresh run, full reward set
  python train.py --reward minimal --iters 300      # the doc's minimal reward, for comparison
  python train.py --dr --resume logs/g1/<run>/model_1500.pt --iters 500   # add domain randomization
"""
import argparse
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner

from g1_walk.config import EnvCfg, train_cfg
from g1_walk.env import G1Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reward", default="full", choices=["full", "minimal"])
    p.add_argument("--dr", action="store_true", help="domain randomization + observation noise")
    p.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)  # leave the other cores to the physics workers
    env_cfg = EnvCfg(num_envs=args.envs, num_workers=args.workers, reward_set=args.reward,
                     domain_rand=args.dr, obs_noise=args.dr)
    env = G1Env(env_cfg, seed=args.seed)

    name = args.name or f"{time.strftime('%Y%m%d_%H%M%S')}_{args.reward}{'_dr' if args.dr else ''}"
    log_dir = os.path.join("logs", "g1", name)
    runner = OnPolicyRunner(env, train_cfg(max_iterations=args.iters), log_dir=log_dir, device="cpu")
    if args.resume:
        runner.load(args.resume)
        print(f"Resumed from {args.resume}")

    try:
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    finally:
        runner.export_policy_to_jit(log_dir)  # policy.pt, loadable by play.py and Unitree's deploy_mujoco
        env.close()
    print(f"Done. Logs and checkpoints in {log_dir}")


if __name__ == "__main__":
    main()
