"""Plot training curves (val/CER per epoch) from TensorBoard logs.

Usage:
    python plot_training_curves.py --runs \
        "100%=/path/to/logs/BiGRU-train" \
        "75%=/path/to/logs/75pct" \
        "50%=/path/to/logs/50pct" \
        "25%=/path/to/logs/25pct"

    python plot_training_curves.py --runs "100%=logs/2026-03-07/BiGRU-train"
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tag: str = "val/CER") -> tuple[list[int], list[float]]:
    """Load scalar values from TensorBoard events."""
    # Try lightning_logs/version_0 first, then root
    import os
    tb_dir = os.path.join(logdir, "lightning_logs", "version_0")
    if not os.path.exists(tb_dir):
        tb_dir = os.path.join(logdir, "lightning_logs")
    if not os.path.exists(tb_dir):
        tb_dir = logdir

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        print(f"  Warning: '{tag}' not found in {tb_dir}")
        print(f"  Available tags: {available}")
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="label=logdir pairs, e.g. '100%%=logs/BiGRU-train'")
    parser.add_argument("--metric", default="val/CER", help="Metric to plot")
    parser.add_argument("--output", default="training_curves.png")
    parser.add_argument("--ylim", type=float, default=100, help="Y-axis upper limit")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    for run in args.runs:
        label, logdir = run.split("=", 1)
        print(f"Loading {label} from {logdir}...")
        steps, values = load_scalars(logdir, args.metric)
        if steps:
            # Convert steps to epochs (steps are global step, need to derive epochs)
            # Plot by index as epoch approximation
            epochs = list(range(len(values)))
            plt.plot(epochs, values, label=label, linewidth=2)
            print(f"  {len(values)} data points, best {args.metric}: {min(values):.2f}% at epoch {epochs[np.argmin(values)]}")
        else:
            print(f"  No data found for {label}")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(args.metric.replace("/", " / "), fontsize=14)
    plt.title(f"{args.metric} vs Epoch — Data Amount Ablation", fontsize=16)
    plt.ylim(0, args.ylim)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
