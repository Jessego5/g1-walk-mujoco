"""Plot reward and episode-length curves from one or more training runs.

  python plot_curves.py logs/g1/run1_full                       # -> results/curves.png
  python plot_curves.py logs/g1/run1_full logs/g1/run2_minimal --out results/compare.png
"""
import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PANELS = [
    ("Train/mean_reward", "Mean episode reward"),
    ("Train/mean_episode_length", "Mean episode length (steps, 50/s)"),
    ("Episode/fall_rate", "Fraction of episodes ending in a fall"),
    ("Episode/rew_tracking_lin_vel", "Velocity-tracking reward per second"),
]


def load(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: ([e.step for e in ea.Scalars(tag)], [e.value for e in ea.Scalars(tag)])
            for tag in ea.Tags()["scalars"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/curves.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for run in args.runs:
        data = load(run)
        for ax, (tag, title) in zip(axes.flat, PANELS):
            if tag in data:
                ax.plot(*data[tag], label=os.path.basename(run.rstrip("/")), linewidth=1.2)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("PPO iteration")
            ax.grid(alpha=0.3)
    if len(args.runs) > 1:
        axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
