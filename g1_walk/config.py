"""All tunable settings in one place.

Values mirror unitree_rl_gym's G1 config (legged_gym/envs/g1/g1_config.py) unless a
comment says otherwise, so results are comparable with the official pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G1_XML = ROOT / "reference/unitree_rl_gym/resources/robots/g1_description/scene.xml"

# MuJoCo joint order (same as the actuator order in g1_12dof.xml and Unitree's deploy_mujoco)
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@dataclass
class EnvCfg:
    # --- parallelism ---
    num_envs: int = 512
    num_workers: int = 8              # CPU processes stepping physics

    # --- timing: 200 Hz physics, 50 Hz policy (same as legged_gym) ---
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0

    # --- robot / control ---
    default_angles: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                             -0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
    kp: tuple = (100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40)
    kd: tuple = (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2)
    action_scale: float = 0.25        # target = default + action_scale * action
    clip_actions: float = 100.0
    init_height: float = 0.8

    # --- observation scaling (matches Unitree deploy config, so policies are interchangeable) ---
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    lin_vel_scale: float = 2.0
    cmd_scale: tuple = (2.0, 2.0, 0.25)
    gait_period: float = 0.8          # seconds, for the sin/cos phase clock

    # --- velocity commands (resampled every resample_s seconds) ---
    resample_s: float = 10.0
    cmd_vx: tuple = (-0.5, 1.0)       # narrower than Unitree's [-1, 1] to make CPU training easier
    cmd_vy: tuple = (-0.3, 0.3)
    cmd_yaw: tuple = (-0.5, 0.5)
    still_prob: float = 0.1           # fraction of envs told to stand still

    # --- termination ---
    min_height: float = 0.4
    max_tilt_cos: float = 0.5         # terminate if body z-axis tilts > 60 deg

    # --- rewards: name -> weight. Every term is multiplied by policy dt, like legged_gym. ---
    reward_set: str = "full"
    tracking_sigma: float = 0.25
    base_height_target: float = 0.78
    soft_dof_pos_limit: float = 0.9
    only_positive_rewards: bool = True

    # --- domain randomization (turn on after basic walking works) ---
    domain_rand: bool = False
    # Unitree randomizes robot friction in [0.1, 1.25] against a ground of 1.0, and Isaac Gym averages the
    # two, so the effective friction there is [0.55, 1.125]. MuJoCo takes the max instead, so we set the
    # effective value directly on every geom.
    friction_range: tuple = (0.55, 1.125)
    added_mass_range: tuple = (-1.0, 3.0)   # kg added to pelvis
    motor_strength_range: tuple = (0.9, 1.1)  # scales kp and kd
    push_interval_s: float = 5.0
    max_push_vel: float = 1.5         # pushes set (not add) the base xy velocity, like legged_gym
    obs_noise: bool = False

    reward_scales: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.reward_scales:
            self.reward_scales = dict(REWARD_SETS[self.reward_set])


REWARD_SETS = {
    # The doc's suggested starting point: track velocity, stay alive, don't waste torque.
    "minimal": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "alive": 0.15,
        "torques": -1e-5,
    },
    # Unitree's G1 reward set. Each extra term fixes a specific failure seen with "minimal":
    "full": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "alive": 0.15,
        "torques": -1e-5,
        "lin_vel_z": -2.0,          # bouncing up and down
        "ang_vel_xy": -0.05,        # wobbling / rolling the torso
        "orientation": -1.0,        # leaning heavily to "fall forward" fast
        "base_height": -10.0,       # crouching very low to stay stable
        "dof_acc": -2.5e-7,         # jittery, high-frequency joint motion
        "dof_vel": -1e-3,
        "action_rate": -0.01,       # actions changing wildly step to step
        "dof_pos_limits": -5.0,     # slamming joints into their limits
        "hip_pos": -1.0,            # splaying legs out with hip roll/yaw
        "contact_no_vel": -0.2,     # feet sliding while on the ground
        "feet_swing_height": -20.0, # shuffling without lifting feet
        "contact": 0.18,            # feet contact matching the gait clock -> alternating steps
    },
}


def train_cfg(experiment: str = "g1", max_iterations: int = 1500) -> dict:
    """rsl_rl 5.x runner config. PPO, GAE and the clipped update all live inside rsl_rl."""
    return {
        "experiment_name": experiment,
        "max_iterations": max_iterations,
        "num_steps_per_env": 24,       # rollout length per env per iteration
        "save_interval": 50,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "logger": "tensorboard",
    }
