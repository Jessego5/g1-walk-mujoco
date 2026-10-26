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
from g1_walk.ppo import load_actor


def load_policy(path, env):
    """Return a function obs(TensorDict) -> action using the policy mean (no exploration noise)."""
    if path == "unitree":
        jit = torch.jit.load(str(ROOT / "reference/unitree_rl_gym/deploy/pre_train/g1/motion.pt"))
    elif path.endswith("policy.pt"):
        jit = torch.jit.load(path)
    else:
        ckpt = torch.load(path, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("custom_ppo"):  # written by train_ppo.py
            jit = load_actor(path, train_cfg())
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
    p.add_argument("--seconds", type=float, default=None, help="default 10 (30 with --scenarios)")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--scenarios", action="store_true",
                   help="random events while walking: new velocity commands (incl. stop) and shoves")
    p.add_argument("--dr", action="store_true",
                   help="randomize friction / pelvis mass / motor strength + sensor noise (re-rolled after a fall)")
    p.add_argument("--event-every", type=float, default=3.0, help="mean seconds between scenario events")
    p.add_argument("--seed", type=int, default=None, help="scenario seed (printed, so a run can be replayed)")
    args = p.parse_args()
    if not (args.checkpoint or args.unitree):
        p.error("give a checkpoint or --unitree")
    seconds = args.seconds or (30.0 if args.scenarios else 10.0)
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 100000)
    rng = np.random.default_rng(seed)

    # one robot, fixed command, never resampled; --dr adds randomization (training's auto-shoves stay off,
    # scenario shoves are applied by hand below)
    cfg = EnvCfg(num_envs=1, num_workers=0, still_prob=0.0, resample_s=1e9, episode_length_s=1e9,
                 domain_rand=args.dr, obs_noise=args.dr, push_interval_s=1e9)
    env = G1Env(cfg, seed=seed)
    policy = load_policy("unitree" if args.unitree else args.checkpoint, env)
    cmd = np.array([args.vx, args.vy, args.yaw])
    env.commands[:] = cmd
    obs = env._observations()

    m, d = env.sim.models[0], env.sim.datas[0]
    steps = int(seconds / env.dt)
    falls, frames, x0 = 0, [], d.qpos[0]

    if args.dr:
        sim = env.sim
        print(f"randomized: friction {m.geom_friction[0, 0]:.2f}, pelvis mass {m.body_mass[sim.pelvis_id] - sim.default_mass:+.1f} kg, "
              f"motor strength x{m.actuator_gainprm[0, 0] / cfg.kp[0]:.2f}")
    FLIGHT_S, START_DIST, END_DIST = 0.6, 2.0, 0.1  # ball: seconds in the air, start / final distance from the pelvis
    balls, hit = [], None
    if args.scenarios:
        print(f"scenario seed {seed} (replay with --seed {seed})")
        shoves, last_event = 0, ("start", 0.0)
        next_event = rng.uniform(0.5, 1.5) * args.event_every

    def scenario_event(t):
        """Pick a random event: a new velocity command (20% chance: stop) or a shove. Returns a description."""
        nonlocal obs
        if rng.random() < 0.5:
            new = np.zeros(3) if rng.random() < 0.2 else np.array(
                [rng.uniform(*cfg.cmd_vx), rng.uniform(*cfg.cmd_vy), rng.uniform(*cfg.cmd_yaw)])
            cmd[:] = new
            env.commands[:] = cmd
            obs = env._observations()
            return "STOP" if not new.any() else f"COMMAND vx={new[0]:+.2f} vy={new[1]:+.2f} yaw={new[2]:+.2f}"
        mag, ang = rng.uniform(0.3, 1.2), rng.uniform(0, 2 * np.pi)
        balls.append({"dir": np.array([np.cos(ang), np.sin(ang)]), "mag": mag, "t0": t})
        return f"THROW ball, will hit at {mag:.2f} m/s, coming from ({-np.cos(ang):+.2f}, {-np.sin(ang):+.2f}) world"

    def draw_overlay(scene, t):
        """Purely visual extras added to the render: the flying ball, the force arrow, the command arrow."""
        p = d.qpos[:3].copy()

        def geom(gtype, size, pos, rgba):
            if scene.ngeom >= scene.maxgeom:
                return None
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            scene.ngeom += 1
            return g

        def arrow(p0, p1, rgba, width):
            g = geom(mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), p0, rgba)
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p1, float))

        # commanded velocity (green): robot-frame command rotated into the world, floating above the pelvis
        qw, qx, qy, qz = d.qpos[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        v = np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]])
        if np.linalg.norm(v) > 0.05:
            top = p + [0, 0, 0.45]
            arrow(top, top + np.append(v, 0) * 0.8, [0.2, 0.9, 0.3, 1], 0.015)
        # flying balls (orange): offset from the pelvis shrinks linearly over the flight, so they home in on the robot
        for b in balls:
            dist = START_DIST + (END_DIST - START_DIST) * min((t - b["t0"]) / FLIGHT_S, 1.0)
            geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.1, 0, 0], p + np.append(-b["dir"] * dist, 0.1), [1, 0.5, 0.1, 1])
        # force arrow (red): points into the robot from the side the ball came from, longer = harder hit
        if hit and t - hit["t"] < 0.6:
            tip = p + np.append(-hit["dir"] * 0.15, 0.1)
            arrow(tip - np.append(hit["dir"], 0) * (0.3 + 0.6 * hit["mag"]), tip, [1, 0.1, 0.1, 1], 0.03)

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
            t = k * env.dt
            if args.scenarios and t >= next_event:
                desc = scenario_event(t)
                last_event = (desc, t)
                print(f"t={t:5.1f}s  {desc}")
                next_event = t + rng.uniform(0.5, 1.5) * args.event_every
            for b in [b for b in balls if t - b["t0"] >= FLIGHT_S]:  # ball arrives: apply the kick
                balls.remove(b)
                env.sim.push(np.array([0]), (b["mag"] * b["dir"])[None])  # sets base xy velocity, like training
                shoves += 1
                hit = {**b, "t": t}
                last_event = (f"HIT {b['mag']:.2f} m/s", t)
                print(f"t={t:5.1f}s  HIT    {b['mag']:.2f} m/s toward ({b['dir'][0]:+.2f}, {b['dir'][1]:+.2f}) world")
            obs, _, done, extras = env.step(policy(obs))
            if done[0]:
                fell = not extras["time_outs"][0]
                falls += int(fell)
                env.commands[:] = cmd
                if args.scenarios:
                    obs = env._observations()
                    if fell:
                        print(f"t={t:5.1f}s  FELL ({t - last_event[1]:.1f}s after: {last_event[0]}) -> respawned standing")
            if args.viewer:
                if not viewer.is_running():
                    break
                if args.scenarios:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        draw_overlay(viewer.user_scn, t)
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.time() - t0)))
            elif k % 2 == 0:  # 25 fps video
                renderer.update_scene(d, cam)
                if args.scenarios:
                    draw_overlay(renderer.scene, t)
                frames.append(renderer.render())

    dist = d.qpos[0] - x0
    if args.scenarios:
        print(f"{seconds:.0f}s of random scenarios: {shoves} shoves, {falls} falls")
    else:
        print(f"{seconds:.0f}s at command vx={args.vx}: moved {dist:.2f} m forward "
              f"(avg {dist / seconds:.2f} m/s), falls={falls}")
    if not args.viewer:
        out = args.out or (os.path.join(os.path.dirname(args.checkpoint), "walk.mp4") if args.checkpoint
                           else "unitree_pretrained.mp4")
        imageio.mimsave(out, frames, fps=25)
        print(f"Saved video to {out}")


if __name__ == "__main__":
    main()
