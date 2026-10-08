"""MuJoCo physics for a batch of G1 robots, split across CPU worker processes.

This layer only knows about physics: set PD targets, step, report state. All RL logic
(observations, rewards, resets, commands) lives in env.py and works on whole batches.
"""
import copy
import multiprocessing as mp

import mujoco
import numpy as np

from .config import FOOT_BODIES, G1_XML, JOINT_NAMES


def load_model(kp, kd, sim_dt, visual=True):
    """Load the G1 and convert its torque motors into PD position actuators.

    MuJoCo then computes tau = kp * (target - q) - kd * dq inside each physics substep,
    clamped to the joint torque limits from the XML, exactly like Unitree's pd_control().

    visual=False drops the mesh geoms. They have no mass or collision (density=0, contype=0),
    so physics is identical, but each copy shrinks from ~250k mesh vertices to almost nothing.
    """
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    if not visual:
        for g in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
            spec.delete(g)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # treats kd implicitly -> stable at 5 ms
    for i, name in enumerate(JOINT_NAMES):
        assert m.actuator(i).name == name, (m.actuator(i).name, name)
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[i, 0] = kp[i]
        m.actuator_biasprm[i, :3] = (0.0, -kp[i], -kd[i])
    return m


class LocalPhysics:
    """A batch of envs simulated in this process. Used inside each worker, and directly by play.py
    (with visual=True so the robot can be rendered)."""

    def __init__(self, n, kp, kd, sim_dt, decimation, visual=False):
        base = load_model(kp, kd, sim_dt, visual=visual)
        self.base = base
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)
        self.models = [copy.deepcopy(base) for _ in range(n)]  # one per env for domain randomization
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.decimation = decimation
        self.foot_ids = [mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        self.pelvis_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.default_mass = base.body_mass[self.pelvis_id]
        self.default_friction = base.geom_friction[:, 0].copy()
        # geom id -> which foot it belongs to (-1 = not a foot)
        self.geom_foot = np.full(base.ngeom, -1)
        for f, b in enumerate(self.foot_ids):
            self.geom_foot[base.geom_bodyid == b] = f
        self._vel = np.zeros(6)

    def step(self, targets):
        for m, d, t in zip(self.models, self.datas, targets):
            d.ctrl[:] = t
            mujoco.mj_step(m, d, nstep=self.decimation)
        return self.state()

    def state(self):
        n = len(self.datas)
        out = {
            "qpos": np.empty((n, 19)), "qvel": np.empty((n, 18)), "tau": np.empty((n, 12)),
            "foot_pos": np.empty((n, 2, 3)), "foot_vel": np.empty((n, 2, 3)),
            "foot_contact": np.zeros((n, 2), bool),
        }
        for i, (m, d) in enumerate(zip(self.models, self.datas)):
            out["qpos"][i] = d.qpos
            out["qvel"][i] = d.qvel
            out["tau"][i] = d.actuator_force
            for f, b in enumerate(self.foot_ids):
                out["foot_pos"][i, f] = d.xpos[b]
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
                out["foot_vel"][i, f] = self._vel[3:]
            if d.ncon:
                feet = self.geom_foot[d.contact.geom[: d.ncon]].ravel()
                out["foot_contact"][i] = [(feet == 0).any(), (feet == 1).any()]
        return out

    def reset(self, ids, qpos, qvel, params):
        for k, i in enumerate(ids):
            m, d = self.models[i], self.datas[i]
            mujoco.mj_resetData(m, d)
            if params is not None:
                friction, added_mass, strength = params[k]
                # MuJoCo uses the max friction of the two touching geoms, so set every geom, not just the floor
                m.geom_friction[:, 0] = friction
                m.body_mass[self.pelvis_id] = self.default_mass + added_mass
                m.actuator_gainprm[:12, 0] = self.kp * strength
                m.actuator_biasprm[:12, 1] = -self.kp * strength
                m.actuator_biasprm[:12, 2] = -self.kd * strength
            d.qpos[:] = qpos[k]
            d.qvel[:] = qvel[k]
            d.ctrl[:] = qpos[k][7:]
            mujoco.mj_forward(m, d)
        return self.state()

    def push(self, ids, dv):
        for k, i in enumerate(ids):
            self.datas[i].qvel[:2] = dv[k]

    def close(self):
        pass


def _worker(pipe, n, kp, kd, sim_dt, decimation):
    batch = LocalPhysics(n, kp, kd, sim_dt, decimation)
    while True:
        cmd, *args = pipe.recv()
        if cmd == "step":
            pipe.send(batch.step(*args))
        elif cmd == "reset":
            pipe.send(batch.reset(*args))
        elif cmd == "push":
            batch.push(*args)
        elif cmd == "close":
            break


class ParallelPhysics:
    """Splits num_envs robots across worker processes; same API as a single batch."""

    def __init__(self, num_envs, num_workers, kp, kd, sim_dt, decimation):
        num_workers = max(1, min(num_workers, num_envs))
        self.splits = np.array_split(np.arange(num_envs), num_workers)
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for idx in self.splits:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, len(idx), kp, kd, sim_dt, decimation), daemon=True)
            p.start()
            self.pipes.append(parent)
            self.procs.append(p)

    def _gather(self):
        parts = [p.recv() for p in self.pipes]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def step(self, targets):
        for pipe, idx in zip(self.pipes, self.splits):
            pipe.send(("step", targets[idx]))
        return self._gather()

    def _route(self, ids):
        """Map global env ids -> (worker, local ids, positions in `ids`)."""
        for w, idx in enumerate(self.splits):
            mask = (ids >= idx[0]) & (ids <= idx[-1])
            if mask.any():
                yield w, ids[mask] - idx[0], np.nonzero(mask)[0]

    def reset(self, ids, qpos, qvel, params=None):
        """Reset some envs; returns the full state of all envs afterwards."""
        touched = set()
        for w, local, pos in self._route(ids):
            self.pipes[w].send(("reset", local, qpos[pos], qvel[pos], None if params is None else params[pos]))
            touched.add(w)
        for w, pipe in enumerate(self.pipes):
            if w not in touched:
                pipe.send(("reset", [], None, None, None))
        return self._gather()

    def push(self, ids, dv):
        for w, local, pos in self._route(ids):
            self.pipes[w].send(("push", local, dv[pos]))

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close",))
        for p in self.procs:
            p.join(timeout=2)
