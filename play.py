"""Deterministic rollout of a trained policy: record a video, or watch live.

  python play.py logs/g1/<run>/model_3000.pt                 # writes <run>/walk.mp4
  python play.py logs/g1/<run>/model_3000.pt --vx 0.8 --seconds 15
  ./view.sh logs/g1/<run>/model_3000.pt                      # live viewer (wraps mjpython)
  python play.py --unitree                                   # Unitree's pretrained policy, as a reference
"""
import argparse
import os
import time

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from g1_walk.config import ROOT, EnvCfg, train_cfg
from g1_walk.env import G1Env


def load_policy(path, env):
    """Return a function obs(TensorDict) -> action using the policy mean (no exploration noise)."""
    if path == "unitree":
        jit = torch.jit.load(str(ROOT / "reference/unitree_rl_gym/deploy/pre_train/g1/motion.pt"))
    elif path.endswith("policy.pt"):
        jit = torch.jit.load(path)
    else:
        runner = OnPolicyRunner(env, train_cfg(), log_dir=None, device="cpu")
        runner.load(path)
        jit = runner.alg.get_policy().as_jit()
    jit.eval()
    return lambda obs: jit(obs["policy"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", help="model_XXX.pt checkpoint or exported policy.pt")
    p.add_argument("--unitree", action="store_true", help="play Unitree's pretrained policy instead")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not (args.checkpoint or args.unitree):
        p.error("give a checkpoint or --unitree")

    # one robot, no randomization, fixed command, never resampled
    cfg = EnvCfg(num_envs=1, num_workers=0, still_prob=0.0, resample_s=1e9, episode_length_s=1e9)
    env = G1Env(cfg)
    policy = load_policy("unitree" if args.unitree else args.checkpoint, env)
    cmd = np.array([args.vx, args.vy, args.yaw])
    env.commands[:] = cmd
    obs = env._observations()

    m, d = env.sim.models[0], env.sim.datas[0]
    steps = int(args.seconds / env.dt)
    falls, frames, x0 = 0, [], d.qpos[0]

    if args.viewer:
        viewer = mujoco.viewer.launch_passive(m, d)
    else:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -15

    with torch.no_grad():
        for k in range(steps):
            t0 = time.time()
            obs, _, done, extras = env.step(policy(obs))
            if done[0]:
                falls += int(not extras["time_outs"][0])
                env.commands[:] = cmd
            if args.viewer:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.time() - t0)))
            elif k % 2 == 0:  # 25 fps video
                renderer.update_scene(d, cam)
                frames.append(renderer.render())

    dist = d.qpos[0] - x0
    print(f"{args.seconds:.0f}s at command vx={args.vx}: moved {dist:.2f} m forward "
          f"(avg {dist / args.seconds:.2f} m/s), falls={falls}")
    if not args.viewer:
        out = args.out or (os.path.join(os.path.dirname(args.checkpoint), "walk.mp4") if args.checkpoint
                           else "unitree_pretrained.mp4")
        imageio.mimsave(out, frames, fps=25)
        print(f"Saved video to {out}")


if __name__ == "__main__":
    main()
