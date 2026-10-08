"""G1 walking task as an rsl_rl VecEnv.

One call to step():
  action (from policy) -> PD joint targets -> 4 physics substeps -> new state
  -> reward terms -> termination check -> reset fallen/timed-out robots -> observation
"""
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .config import EnvCfg
from .physics import LocalPhysics, ParallelPhysics


def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v (N,3) into the body frame given MuJoCo quats q (N,4) = (w,x,y,z)."""
    w, u = q[:, :1], q[:, 1:]
    return (v * (2 * w**2 - 1) - np.cross(u, v) * w * 2 + u * np.sum(u * v, axis=1, keepdims=True) * 2)


class G1Env(VecEnv):
    def __init__(self, cfg: EnvCfg, seed: int = 0):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = 12
        self.device = "cpu"
        self.dt = cfg.sim_dt * cfg.decimation
        self.max_episode_length = int(cfg.episode_length_s / self.dt)
        self.rng = np.random.default_rng(seed)

        self.default = np.array(cfg.default_angles)
        if cfg.num_workers == 0:  # single process with meshes, for play/rendering
            self.sim = LocalPhysics(cfg.num_envs, cfg.kp, cfg.kd, cfg.sim_dt, cfg.decimation, visual=True)
        else:
            self.sim = ParallelPhysics(cfg.num_envs, cfg.num_workers, cfg.kp, cfg.kd, cfg.sim_dt, cfg.decimation)
        self.joint_range = self._joint_limits()

        n = self.num_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.actions = np.zeros((n, 12))
        self.last_actions = np.zeros((n, 12))
        self.last_dq = np.zeros((n, 12))
        self.commands = np.zeros((n, 3))
        self.ep_sums = {k: np.zeros(n) for k in cfg.reward_scales}
        self.push_every = int(cfg.push_interval_s / self.dt)
        self.resample_every = int(cfg.resample_s / self.dt)
        self.step_count = 0

        self.state = self.sim.reset(np.arange(n), *self._initial_state(n), self._dr_params(n))
        self._resample_commands(np.arange(n))
        self._update_derived()
        self.obs = self._observations()

    # ------------------------------------------------------------------ helpers
    def _joint_limits(self):
        from .physics import load_model
        m = load_model(self.cfg.kp, self.cfg.kd, self.cfg.sim_dt)
        r = m.jnt_range[1:13].copy()  # joint 0 is the floating base
        mid, half = r.mean(1), (r[:, 1] - r[:, 0]) / 2 * self.cfg.soft_dof_pos_limit
        return np.stack([mid - half, mid + half], 1)

    def _initial_state(self, k):
        qpos = np.zeros((k, 19))
        qpos[:, 2] = self.cfg.init_height
        qpos[:, 3] = 1.0
        qpos[:, 7:] = self.default + self.rng.uniform(-0.1, 0.1, (k, 12))
        qvel = np.zeros((k, 18))
        qvel[:, :6] = self.rng.uniform(-0.2, 0.2, (k, 6))
        return qpos, qvel

    def _dr_params(self, k):
        c = self.cfg
        if not c.domain_rand:
            return None
        return np.stack([
            self.rng.uniform(*c.friction_range, k),
            self.rng.uniform(*c.added_mass_range, k),
            self.rng.uniform(*c.motor_strength_range, k),
        ], 1)

    def _resample_commands(self, ids):
        c, k = self.cfg, len(ids)
        if k == 0:
            return
        cmd = np.stack([self.rng.uniform(*c.cmd_vx, k), self.rng.uniform(*c.cmd_vy, k),
                        self.rng.uniform(*c.cmd_yaw, k)], 1)
        cmd[np.linalg.norm(cmd[:, :2], axis=1) < 0.2, :2] = 0.0  # tiny commands -> stand still
        cmd[self.rng.random(k) < c.still_prob] = 0.0
        self.commands[ids] = cmd

    def _update_derived(self):
        s = self.state
        quat = s["qpos"][:, 3:7]
        self.q = s["qpos"][:, 7:]
        self.dq = s["qvel"][:, 6:]
        self.base_lin_vel = quat_rotate_inverse(quat, s["qvel"][:, :3])
        self.base_ang_vel = s["qvel"][:, 3:6]  # MuJoCo free-joint angular velocity is already in the body frame
        self.gravity = quat_rotate_inverse(quat, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        self.height = s["qpos"][:, 2]
        t = self.episode_length_buf.numpy() * self.dt
        self.phase = (t % self.cfg.gait_period) / self.cfg.gait_period
        self.leg_phase = np.stack([self.phase, (self.phase + 0.5) % 1.0], 1)  # legs half a cycle apart

    # ------------------------------------------------------------- observations
    def _observations(self):
        c = self.cfg
        clock = np.stack([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], 1)
        policy = np.concatenate([
            self.base_ang_vel * c.ang_vel_scale,        # 3   how fast the torso is rotating
            self.gravity,                               # 3   which way is down (= torso tilt)
            self.commands * np.array(c.cmd_scale),      # 3   desired vx, vy, yaw rate
            (self.q - self.default) * c.dof_pos_scale,  # 12  joint angles
            self.dq * c.dof_vel_scale,                  # 12  joint velocities
            self.actions,                               # 12  previous action
            clock,                                      # 2   gait clock
        ], 1)                                           # = 47, same layout as Unitree's deploy script
        if c.obs_noise:
            noise = np.concatenate([np.full(3, 0.2 * c.ang_vel_scale), np.full(3, 0.05), np.zeros(3),
                                    np.full(12, 0.01), np.full(12, 1.5 * c.dof_vel_scale), np.zeros(14)])
            policy = policy + self.rng.uniform(-1, 1, policy.shape) * noise
        # The critic also gets the true base velocity, which a real robot can't measure directly.
        privileged = self.base_lin_vel * c.lin_vel_scale
        return TensorDict({
            "policy": torch.as_tensor(policy, dtype=torch.float32),
            "privileged": torch.as_tensor(privileged, dtype=torch.float32),
        }, batch_size=[self.num_envs])

    def get_observations(self):
        return self.obs

    # ------------------------------------------------------------------ rewards
    def _reward_terms(self):
        c, s = self.cfg, self.state
        contact = s["foot_contact"]
        stance = self.leg_phase < 0.55
        lin_err = np.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, 1)
        yaw_err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        below = np.clip(self.joint_range[:, 0] - self.q, 0, None)
        above = np.clip(self.q - self.joint_range[:, 1], 0, None)
        return {
            "tracking_lin_vel": np.exp(-lin_err / c.tracking_sigma),
            "tracking_ang_vel": np.exp(-yaw_err / c.tracking_sigma),
            "alive": np.ones(self.num_envs),
            "torques": np.sum(s["tau"] ** 2, 1),
            "lin_vel_z": self.base_lin_vel[:, 2] ** 2,
            "ang_vel_xy": np.sum(self.base_ang_vel[:, :2] ** 2, 1),
            "orientation": np.sum(self.gravity[:, :2] ** 2, 1),
            "base_height": (self.height - c.base_height_target) ** 2,
            "dof_acc": np.sum(((self.last_dq - self.dq) / self.dt) ** 2, 1),
            "dof_vel": np.sum(self.dq**2, 1),
            "action_rate": np.sum((self.last_actions - self.actions) ** 2, 1),
            "dof_pos_limits": np.sum(below + above, 1),
            "hip_pos": np.sum(self.q[:, [1, 2, 7, 8]] ** 2, 1),  # hip roll + yaw, both legs
            "contact_no_vel": np.sum((s["foot_vel"] * contact[..., None]) ** 2, (1, 2)),
            "feet_swing_height": np.sum((s["foot_pos"][..., 2] - 0.08) ** 2 * ~contact, 1),
            "contact": np.sum(contact == stance, 1).astype(float),
        }

    def _compute_reward(self):
        terms = self._reward_terms()
        total = np.zeros(self.num_envs)
        for name, scale in self.cfg.reward_scales.items():
            r = terms[name] * scale * self.dt
            self.ep_sums[name] += r
            total += r
        if self.cfg.only_positive_rewards:
            total = np.clip(total, 0, None)
        return total

    # --------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        c = self.cfg
        self.last_actions = self.actions
        self.last_dq = self.dq
        self.actions = np.clip(actions.cpu().numpy().astype(np.float64), -c.clip_actions, c.clip_actions)
        targets = self.default + c.action_scale * self.actions

        self.state = self.sim.step(targets)
        self.episode_length_buf += 1
        self.step_count += 1
        self._update_derived()

        reward = self._compute_reward()
        fell = (self.height < c.min_height) | (-self.gravity[:, 2] < c.max_tilt_cos)
        timeout = self.episode_length_buf.numpy() >= self.max_episode_length
        done = fell | timeout

        extras = {"time_outs": torch.as_tensor(timeout & ~fell)}
        ids = np.nonzero(done)[0]
        if len(ids):
            extras["episode"] = {f"rew_{k}": float(v[ids].mean() / c.episode_length_s)
                                 for k, v in self.ep_sums.items()}
            extras["episode"]["fall_rate"] = float(fell[ids].mean())
            for v in self.ep_sums.values():
                v[ids] = 0.0
            self.state = self.sim.reset(ids, *self._initial_state(len(ids)), self._dr_params(len(ids)))
            self.episode_length_buf[ids] = 0
            self.actions[ids] = 0.0
            self._resample_commands(ids)
            self._update_derived()
            self.last_dq[ids] = self.dq[ids]
            self.last_actions[ids] = 0.0

        resample = np.nonzero((self.episode_length_buf.numpy() % self.resample_every) == 0)[0]
        self._resample_commands(np.setdiff1d(resample, ids))
        if c.domain_rand and self.step_count % self.push_every == 0:
            self.sim.push(np.arange(self.num_envs), self.rng.uniform(-c.max_push_vel, c.max_push_vel, (self.num_envs, 2)))

        self.obs = self._observations()
        return self.obs, torch.as_tensor(reward, dtype=torch.float32), torch.as_tensor(done, dtype=torch.long), extras

    def close(self):
        self.sim.close()
