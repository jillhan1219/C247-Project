"""Decode test set examples and print predicted vs ground truth text.

Usage:
    python decode_examples.py --decoder greedy
    python decode_examples.py --decoder beam
    python decode_examples.py --decoder beam --checkpoint /path/to/ckpt
    python decode_examples.py --decoder beam --model tds_conv_ctc
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from collections import Counter

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.data import LabelData
from emg2qwerty.charset import charset

parser = argparse.ArgumentParser()
parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
parser.add_argument("--checkpoint", default="/home/ubuntu/jinru/emg2qwerty/logs/2026-03-07/BiGRU-train/checkpoints/best.ckpt")
parser.add_argument("--model", default="bigru_ctc", help="Model config name (e.g., bigru_ctc, tds_conv_ctc)")
args = parser.parse_args()

PROJECT_ROOT = "/home/ubuntu/jinru/emg2qwerty"
CONFIG_DIR = f"{PROJECT_ROOT}/config"

# Load config via Hydra compose API
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

print(f"Decoder: {args.decoder} | Model: {args.model} | Checkpoint: {args.checkpoint}")

# Helper functions (same as train.py)
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

# Load test data with window_length=None for full session
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
_charset = charset()

# Track per-character errors
substitutions = Counter()  # (gt_char, pred_char) -> count
deletions = Counter()       # gt_char -> count
insertions = Counter()      # pred_char -> count

all_examples = []

for batch_idx, batch in enumerate(test_dl):
    inputs = batch["inputs"].cuda()
    targets = batch["targets"]
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]
    N = len(input_lengths)

    with torch.no_grad():
        emissions = module(inputs)

    T_diff = inputs.shape[0] - emissions.shape[0]
    emission_lengths = (input_lengths - T_diff).cpu().numpy()

    predictions = decoder.decode_batch(
        emissions=emissions.cpu().numpy(),
        emission_lengths=emission_lengths,
    )

    targets_np = targets.cpu().numpy()
    target_lengths_np = target_lengths.cpu().numpy()

    for i in range(N):
        target = LabelData.from_labels(targets_np[:target_lengths_np[i], i])
        pred = predictions[i]
        all_examples.append((target.text, pred.text))

# Print examples
for idx, (gt, pred) in enumerate(all_examples):
    print(f"=== Session {idx} ===")
    print(f"GT  : {gt[:300]}")
    print(f"PRED: {pred[:300]}")
    print()

# Simple edit-distance-based error analysis
import difflib
for gt, pred in all_examples:
    sm = difflib.SequenceMatcher(None, gt, pred)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            for g, p in zip(gt[i1:i2], pred[j1:j2]):
                substitutions[(g, p)] += 1
            # Handle length mismatch
            extra_gt = (i2 - i1) - (j2 - j1)
            if extra_gt > 0:
                for g in gt[i2 - extra_gt:i2]:
                    deletions[g] += 1
            elif extra_gt < 0:
                for p in pred[j2 + extra_gt:j2]:
                    insertions[p] += 1
        elif op == "delete":
            for g in gt[i1:i2]:
                deletions[g] += 1
        elif op == "insert":
            for p in pred[j1:j2]:
                insertions[p] += 1

print("\n=== Top 15 Substitutions (GT -> PRED) ===")
for (g, p), count in substitutions.most_common(15):
    print(f"  '{g}' -> '{p}': {count}")

print("\n=== Top 10 Deletions (missed characters) ===")
for ch, count in deletions.most_common(10):
    print(f"  '{ch}': {count}")

print("\n=== Top 10 Insertions (extra characters) ===")
for ch, count in insertions.most_common(10):
    print(f"  '{ch}': {count}")
