"""Analyze CER vs effective sampling rate (temporal downsampling at inference).

The spectrogram output is at 125Hz (2kHz EMG / hop_length=16). We simulate
lower sampling rates by taking every Nth frame at inference time.

Usage:
    python sampling_rate_analysis.py
    python sampling_rate_analysis.py --checkpoint /path/to/ckpt --model bigru_ctc
"""

import argparse
import torch
import numpy as np
import Levenshtein
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
args = parser.parse_args()

PROJECT_ROOT = "/home/ubuntu/jinru/emg2qwerty"
CONFIG_DIR = f"{PROJECT_ROOT}/config"

# Downsample factors to test
# Original spectrogram rate = 2000 / hop_length(16) = 125 Hz
BASE_RATE = 125.0
DOWNSAMPLE_FACTORS = [1, 2, 3, 4, 6, 8]

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


def run_test_with_downsample(factor: int) -> float:
    """Run test with temporal downsampling and return CER."""
    total_edits = 0
    total_target_len = 0

    for batch in test_dl:
        inputs = batch["inputs"].cuda()  # (T, N, 2, 16, 33)
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        # Temporal downsampling: take every `factor`-th frame
        if factor > 1:
            ds_inputs = inputs[::factor]
            ds_input_lengths = torch.div(input_lengths, factor, rounding_mode='floor')
        else:
            ds_inputs = inputs
            ds_input_lengths = input_lengths

        with torch.no_grad():
            emissions = module(ds_inputs)

        T_diff = ds_inputs.shape[0] - emissions.shape[0]
        emission_lengths = (ds_input_lengths - T_diff).clamp(min=1).cpu().numpy()

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


# Run experiments
print(f"{'Factor':>8} | {'Eff. Rate (Hz)':>15} | {'Test CER':>10}")
print("-" * 45)

results = []
for factor in DOWNSAMPLE_FACTORS:
    eff_rate = BASE_RATE / factor
    cer = run_test_with_downsample(factor)
    print(f"{factor:>8} | {eff_rate:>13.1f}Hz | {cer:>8.2f}%")
    results.append({"factor": factor, "rate": eff_rate, "cer": cer})

# Summary table
print(f"\n{'=' * 60}")
print("SUMMARY TABLE (for results.md)")
print(f"{'=' * 60}")
print("| Downsample Factor | Effective Rate (Hz) | Test CER |")
print("|-------------------|--------------------:|----------|")
for r in results:
    print(f"| {r['factor']}x | {r['rate']:.1f} | {r['cer']:.2f}% |")
