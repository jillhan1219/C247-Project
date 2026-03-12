"""Analyze CER vs number of active electrode channels.

Masks out channels at inference time (no retraining) to measure how many
electrodes are needed for good decoding performance.

Usage:
    python channel_analysis.py
    python channel_analysis.py --checkpoint /path/to/ckpt --model bigru_ctc
    python channel_analysis.py --n_trials 5
"""

import argparse
import torch
import numpy as np
import Levenshtein
from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.data import LabelData
from emg2qwerty.charset import charset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="/home/ubuntu/jinru/emg2qwerty/logs/2026-03-07/BiGRU-train/checkpoints/best.ckpt")
parser.add_argument("--model", default="bigru_ctc")
parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
parser.add_argument("--n_trials", type=int, default=3, help="Number of random trials per channel count")
args = parser.parse_args()

PROJECT_ROOT = "/home/ubuntu/jinru/emg2qwerty"
CONFIG_DIR = f"{PROJECT_ROOT}/config"

# Channel counts to test (out of 16 per band)
CHANNEL_COUNTS = [16, 14, 12, 10, 8, 6, 4, 2]

# Load config
OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task, replace=True)
overrides = [
    "user=single_user",
    f"model={args.model}",
    f"decoder=ctc_{args.decoder}",
    f"dataset.root={PROJECT_ROOT}/data",
]
if args.decoder == "beam":
    overrides.append(f"decoder.lm_path={PROJECT_ROOT}/models/lm/wikitext-103-6gram-charlm.bin")
with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    config = compose(config_name="base", overrides=overrides)


def _full_session_paths(dataset):
    sessions = [session["session"] for session in dataset]
    return [Path(config.dataset.root).joinpath(f"{session}.hdf5") for session in sessions]

def _build_transform(configs):
    return transforms.Compose([instantiate(cfg) for cfg in configs])


# Load model
module = instantiate(
    config.module,
    optimizer=config.optimizer,
    lr_scheduler=config.lr_scheduler,
    decoder=config.decoder,
    _recursive_=False,
).load_from_checkpoint(args.checkpoint)
module.eval()
module.cuda()

# Load test data
datamodule = instantiate(
    config.datamodule,
    batch_size=config.batch_size,
    num_workers=1,
    train_sessions=_full_session_paths(config.dataset.train),
    val_sessions=_full_session_paths(config.dataset.val),
    test_sessions=_full_session_paths(config.dataset.test),
    train_transform=_build_transform(config.transforms.train),
    val_transform=_build_transform(config.transforms.val),
    test_transform=_build_transform(config.transforms.test),
    window_length=None,
    padding=[0, 0],
    _convert_="object",
)
datamodule.setup("test")
test_dl = datamodule.test_dataloader()

decoder = instantiate(config.decoder)


def run_test_with_mask(channel_mask: torch.Tensor) -> float:
    """Run test set with given channel mask and return CER (same as official metric).

    Args:
        channel_mask: Boolean tensor of shape (16,), True = keep, False = zero out.
                      Applied to both bands identically.
    """
    total_edits = 0
    total_target_len = 0
    for batch in test_dl:
        inputs = batch["inputs"].cuda()  # (T, N, 2, 16, 33)
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        # Apply channel mask: zero out masked channels in both bands
        masked_inputs = inputs.clone()
        mask = channel_mask.to(inputs.device)  # (16,)
        masked_inputs[:, :, :, ~mask, :] = 0.0

        with torch.no_grad():
            emissions = module(masked_inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = (input_lengths - T_diff).cpu().numpy()

        predictions = decoder.decode_batch(
            emissions=emissions.cpu().numpy(),
            emission_lengths=emission_lengths,
        )

        targets_np = targets.cpu().numpy()
        target_lengths_np = target_lengths.cpu().numpy()

        for i in range(N):
            gt = LabelData.from_labels(targets_np[:target_lengths_np[i], i]).text
            pred = predictions[i].text
            editops = Levenshtein.editops(pred, gt)
            total_edits += len(editops)
            total_target_len += len(gt)

    return total_edits / max(total_target_len, 1) * 100


# --- Importance ranking via activation magnitude ---
print("Computing channel importance via input activation magnitude...")
importance = torch.zeros(16)
n_samples = 0
for batch in test_dl:
    inputs = batch["inputs"].cuda()  # (T, N, 2, 16, 33)
    # Average absolute activation per channel across T, N, bands, freq
    channel_act = inputs.abs().mean(dim=(0, 1, 2, 4))  # (16,)
    importance += channel_act.cpu() * inputs.shape[1]  # weight by batch size
    n_samples += inputs.shape[1]
importance /= n_samples
importance_ranking = importance.argsort(descending=True)
print(f"Channel importance ranking (most→least): {importance_ranking.tolist()}")
print(f"Channel activations: {[f'{importance[i]:.4f}' for i in importance_ranking.tolist()]}")

# --- Run experiments ---
print(f"\nTesting {len(CHANNEL_COUNTS)} channel counts × {args.n_trials} random trials + importance-based...")
print(f"{'Channels':>10} | {'Random CER (mean±std)':>25} | {'Importance CER':>15}")
print("-" * 60)

results = []
for n_channels in CHANNEL_COUNTS:
    # Importance-based: keep top-n most important channels
    imp_mask = torch.zeros(16, dtype=torch.bool)
    imp_mask[importance_ranking[:n_channels]] = True
    imp_cer = run_test_with_mask(imp_mask)

    # Random dropping: average over n_trials
    if n_channels == 16:
        rand_cers = [imp_cer]  # All channels = same result
    else:
        rand_cers = []
        for trial in range(args.n_trials):
            rand_indices = torch.randperm(16)[:n_channels]
            rand_mask = torch.zeros(16, dtype=torch.bool)
            rand_mask[rand_indices] = True
            rand_cer = run_test_with_mask(rand_mask)
            rand_cers.append(rand_cer)

    rand_mean = np.mean(rand_cers)
    rand_std = np.std(rand_cers)

    print(f"{n_channels:>10} | {rand_mean:>18.2f}% ± {rand_std:.2f}% | {imp_cer:>12.2f}%")
    results.append({
        "n_channels": n_channels,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "importance_cer": imp_cer,
    })

# Print summary table for results.md
print(f"\n{'=' * 60}")
print("SUMMARY TABLE (for results.md)")
print(f"{'=' * 60}")
print(f"| Channels | Random CER (mean±std) | Importance-based CER |")
print(f"|----------|----------------------|---------------------|")
for r in results:
    print(f"| {r['n_channels']:>8} | {r['random_mean']:>14.2f}% ± {r['random_std']:.2f}% | {r['importance_cer']:>17.2f}% |")

print(f"\nChannel importance ranking (most→least important): {importance_ranking.tolist()}")
