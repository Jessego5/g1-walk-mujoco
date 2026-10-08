"""Measure gait quality of one or more policies, to compare reward sets with numbers.

  python gait_stats.py logs/g1/run1_full/model_600.pt logs/g1/run2_minimal/model_599.pt

All robots get the same command (default 0.5 m/s forward) for 10 s, no randomization.
"""
import argparse

import numpy as np
import torch

from g1_walk.config import EnvCfg
from g1_walk.env import G1Env
from play import load_policy


def measure(ckpt, vx, envs, seconds):
    cfg = EnvCfg(num_envs=envs, num_workers=8, still_prob=0.0, resample_s=1e9)
    env = G1Env(cfg, seed=1)
    policy = load_policy(ckpt, env)
    env.commands[:] = [vx, 0.0, 0.0]
    obs = env._observations()
    keys = ["speed", "lean", "height", "air_z", "down_z", "slip", "jitter", "both_air", "both_down", "torque"]
    log = {k: [] for k in keys}
    fell = np.zeros(envs, bool)
    with torch.no_grad():
        for _ in range(int(seconds / env.dt)):
            prev = env.actions.copy()
            obs, _, done, extras = env.step(policy(obs))
            fell |= done.numpy().astype(bool) & ~extras["time_outs"].numpy()
            env.commands[:] = [vx, 0.0, 0.0]
            s, c = env.state, env.state["foot_contact"]
            log["speed"].append(env.base_lin_vel[:, 0])
            log["lean"].append(np.degrees(np.arccos(np.clip(-env.gravity[:, 2], -1, 1))))
            log["height"].append(env.height)
            log["air_z"].append(s["foot_pos"][..., 2][~c])     # foot heights while swinging
            log["down_z"].append(s["foot_pos"][..., 2][c])     # foot heights while planted
            log["slip"].append(np.sum(np.linalg.norm(s["foot_vel"][..., :2], axis=2) * c, 1) / np.maximum(c.sum(1), 1))
            log["jitter"].append(np.abs(env.actions - prev).mean(1))
            log["both_air"].append(~c.any(1))
            log["both_down"].append(c.all(1))
            log["torque"].append(np.abs(s["tau"]).mean(1))
    env.close()
    air, down = np.concatenate(log.pop("air_z")), np.concatenate(log.pop("down_z"))
    out = {k: float(np.mean(v)) for k, v in log.items()}
    out["swing_h"] = float(air.mean() - down.mean()) if len(air) else 0.0  # clearance above planted height
    out["fell"] = float(fell.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()

    rows = [
        ("speed", "forward speed (m/s)", "{:.2f}"),
        ("fell", "fell within 10 s", "{:.0%}"),
        ("lean", "torso tilt (deg)", "{:.1f}"),
        ("height", "pelvis height (m)", "{:.3f}"),
        ("swing_h", "swing-foot clearance (m)", "{:.3f}"),
        ("slip", "planted-foot slip (m/s)", "{:.3f}"),
        ("jitter", "action change per step", "{:.3f}"),
        ("both_air", "time with both feet in air", "{:.0%}"),
        ("both_down", "time with both feet down", "{:.0%}"),
        ("torque", "mean |torque| (N·m)", "{:.1f}"),
    ]
    results = [measure(c, args.vx, args.envs, args.seconds) for c in args.checkpoints]
    names = [c.split("/")[-2] + "/" + c.split("/")[-1].replace(".pt", "") for c in args.checkpoints]
    print(f"\ncommand: {args.vx} m/s forward\n")
    print(f"| {'':28s} | " + " | ".join(f"{n:>26s}" for n in names) + " |")
    print("|" + "-" * 30 + "|" + "|".join("-" * 28 for _ in names) + "|")
    for key, label, fmt in rows:
        print(f"| {label:28s} | " + " | ".join(f"{fmt.format(r[key]):>26s}" for r in results) + " |")


if __name__ == "__main__":
    main()
