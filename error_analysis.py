"""Per-character confusion matrix and left/right hand error analysis.

Usage:
    python error_analysis.py --decoder greedy
    python error_analysis.py --decoder beam
    python error_analysis.py --decoder greedy --checkpoint /path/to/ckpt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from emg2qwerty import transforms, utils
from emg2qwerty.data import LabelData
from emg2qwerty.charset import charset

import Levenshtein

parser = argparse.ArgumentParser()
parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
parser.add_argument("--checkpoint", default="/home/ubuntu/jinru/emg2qwerty/logs/2026-03-07/BiGRU-train/checkpoints/best.ckpt")
parser.add_argument("--model", default="bigru_ctc", help="Model config name")
parser.add_argument("--output_prefix", default="error_analysis", help="Output file prefix")
args = parser.parse_args()

PROJECT_ROOT = "/home/ubuntu/jinru/emg2qwerty"
CONFIG_DIR = f"{PROJECT_ROOT}/config"

# Standard QWERTY left/right hand split (touch typing)
LEFT_HAND_KEYS = set("qwertasdfgzxcvb12345`~!@#$%")
RIGHT_HAND_KEYS = set("yuiophjklnm67890-=[]\\;',./^&*()_+{}|:\"<>?")
# Space, enter, backspace, shift are pressed by both hands — categorize separately
BOTH_HANDS_KEYS = set(" ")  # space
SPECIAL_KEYS = {"⏎", "⌫", "⇧"}

def get_hand(char):
    """Return 'left', 'right', 'both', or 'special' for a character."""
    c = char.lower()
    if char in SPECIAL_KEYS:
        return "special"
    if c in LEFT_HAND_KEYS:
        return "left"
    if c in RIGHT_HAND_KEYS:
        return "right"
    if c in BOTH_HANDS_KEYS:
        return "both"
    return "other"

# ── Load model and data ──
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
    sessions = [s["session"] for s in dataset]
    return [Path(config.dataset.root).joinpath(f"{s}.hdf5") for s in sessions]

def _build_transform(configs):
    return transforms.Compose([instantiate(cfg) for cfg in configs])

module = instantiate(
    config.module,
    optimizer=config.optimizer,
    lr_scheduler=config.lr_scheduler,
    decoder=config.decoder,
    _recursive_=False,
).load_from_checkpoint(args.checkpoint)
module.eval()
module.cuda()

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

# ── Decode test set ──
all_examples = []
for batch in test_dl:
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

print(f"Decoded {len(all_examples)} test sessions with {args.decoder} decoder\n")

# ── Error analysis using Levenshtein editops ──
substitutions = Counter()   # (gt_char, pred_char) -> count
deletions = Counter()        # gt_char -> count
insertions = Counter()       # pred_char -> count
char_total = Counter()       # gt_char -> total occurrences

for gt, pred in all_examples:
    # Count total occurrences of each char in ground truth
    for c in gt:
        char_total[c] += 1

    ops = Levenshtein.editops(gt, pred)
    for op, i, j in ops:
        if op == "replace":
            substitutions[(gt[i], pred[j])] += 1
        elif op == "delete":
            deletions[gt[i]] += 1
        elif op == "insert":
            insertions[pred[j]] += 1

# ── 1. Confusion Matrix Heatmap ──
print("=== Building confusion matrix ===")

# Find the most-confused characters (top substitution pairs)
# Focus on characters with enough errors to be meaningful
all_error_chars = set()
for (g, p), count in substitutions.most_common(100):
    if count >= 3:  # minimum threshold
        all_error_chars.add(g)
        all_error_chars.add(p)

# Also add chars with high deletion/insertion rates
for ch, count in deletions.most_common(20):
    all_error_chars.add(ch)
for ch, count in insertions.most_common(20):
    all_error_chars.add(ch)

# Build display-friendly labels
def display_char(c):
    if c == " ":
        return "⎵"
    if c == "⏎":
        return "⏎"
    if c == "⌫":
        return "⌫"
    if c == "⇧":
        return "⇧"
    return c

# Sort chars: lowercase letters, then digits, then special
def sort_key(c):
    if c.isalpha():
        return (0, c.lower(), c)
    if c.isdigit():
        return (1, c, c)
    return (2, c, c)

sorted_chars = sorted(all_error_chars, key=sort_key)

# Build confusion matrix (rows=GT, cols=PRED)
n = len(sorted_chars)
char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
conf_matrix = np.zeros((n, n), dtype=int)

for (g, p), count in substitutions.items():
    if g in char_to_idx and p in char_to_idx:
        conf_matrix[char_to_idx[g], char_to_idx[p]] += count

# Plot
fig, ax = plt.subplots(figsize=(14, 12))
labels = [display_char(c) for c in sorted_chars]

# Use log scale for better visibility
mask = conf_matrix == 0
sns.heatmap(
    conf_matrix,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    mask=mask,
    ax=ax,
    cbar_kws={"label": "Substitution Count"},
    square=True,
    linewidths=0.5,
)
ax.set_xlabel("Predicted Character", fontsize=12)
ax.set_ylabel("Ground Truth Character", fontsize=12)
ax.set_title(f"Character Substitution Confusion Matrix ({args.decoder} decoding)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{args.output_prefix}_confusion_{args.decoder}.png", dpi=150)
print(f"Saved: {args.output_prefix}_confusion_{args.decoder}.png")

# ── Print top substitution pairs with QWERTY proximity info ──
# QWERTY row/col positions for proximity analysis
QWERTY_POS = {
    'q': (0,0), 'w': (0,1), 'e': (0,2), 'r': (0,3), 't': (0,4),
    'y': (0,5), 'u': (0,6), 'i': (0,7), 'o': (0,8), 'p': (0,9),
    'a': (1,0), 's': (1,1), 'd': (1,2), 'f': (1,3), 'g': (1,4),
    'h': (1,5), 'j': (1,6), 'k': (1,7), 'l': (1,8),
    'z': (2,0), 'x': (2,1), 'c': (2,2), 'v': (2,3), 'b': (2,4),
    'n': (2,5), 'm': (2,6),
}

def qwerty_distance(c1, c2):
    """Manhattan distance on QWERTY layout. Returns None if not both lowercase letters."""
    c1, c2 = c1.lower(), c2.lower()
    if c1 in QWERTY_POS and c2 in QWERTY_POS:
        r1, c1p = QWERTY_POS[c1]
        r2, c2p = QWERTY_POS[c2]
        return abs(r1 - r2) + abs(c1p - c2p)
    return None

print(f"\n=== Top 25 Substitutions (GT → PRED) with QWERTY distance ===")
print(f"{'GT':>4} → {'PRED':<4} {'Count':>6}  {'QWERTY dist':>11}  {'Same hand?':>10}  {'GT hand':>8}  {'PRED hand':>9}")
print("-" * 70)
distances = []
for (g, p), count in substitutions.most_common(25):
    dist = qwerty_distance(g, p)
    gh = get_hand(g)
    ph = get_hand(p)
    same = "yes" if gh == ph and gh in ("left", "right") else ("n/a" if gh in ("both", "special") or ph in ("both", "special") else "no")
    dist_str = str(dist) if dist is not None else "n/a"
    print(f"  '{display_char(g)}' → '{display_char(p)}'  {count:>5}  {dist_str:>11}  {same:>10}  {gh:>8}  {ph:>9}")
    if dist is not None:
        distances.extend([dist] * count)

if distances:
    print(f"\nMean QWERTY distance for letter substitutions: {np.mean(distances):.2f}")
    print(f"Median QWERTY distance: {np.median(distances):.1f}")
    # Compare with random baseline
    letters = [c for c in QWERTY_POS]
    random_dists = []
    for c1 in letters:
        for c2 in letters:
            if c1 != c2:
                random_dists.append(qwerty_distance(c1, c2))
    print(f"Expected random QWERTY distance: {np.mean(random_dists):.2f}")

# ── 2. Left vs Right Hand Error Analysis ──
print(f"\n=== Left Hand vs Right Hand Error Analysis ===")

# Per-hand character counts and errors
hand_totals = defaultdict(int)     # hand -> total GT chars
hand_errors = defaultdict(int)     # hand -> total errors (sub + del)
hand_subs = defaultdict(int)
hand_dels = defaultdict(int)
hand_ins = defaultdict(int)

# Count totals per hand
for ch, count in char_total.items():
    hand = get_hand(ch)
    hand_totals[hand] += count

# Substitutions and deletions are errors on GT characters
for (g, p), count in substitutions.items():
    hand = get_hand(g)
    hand_subs[hand] += count
    hand_errors[hand] += count

for ch, count in deletions.items():
    hand = get_hand(ch)
    hand_dels[hand] += count
    hand_errors[hand] += count

# Insertions are errors on predicted characters
for ch, count in insertions.items():
    hand = get_hand(ch)
    hand_ins[hand] += count

print(f"\n{'Hand':<10} {'Total GT':>10} {'Errors':>8} {'Error%':>8} {'Subs':>8} {'Dels':>8} {'Ins':>8} {'Sub%':>8} {'Del%':>8}")
print("-" * 90)
for hand in ["left", "right", "both", "special"]:
    total = hand_totals[hand]
    if total == 0:
        continue
    errs = hand_errors[hand]
    subs = hand_subs[hand]
    dels = hand_dels[hand]
    ins = hand_ins[hand]
    print(f"{hand:<10} {total:>10} {errs:>8} {errs/total*100:>7.2f}% {subs:>8} {dels:>8} {ins:>8} {subs/total*100:>7.2f}% {dels/total*100:>7.2f}%")

# Per-character error rate for each hand
print(f"\n=== Per-Character Error Rates (Left Hand) ===")
print(f"{'Char':<6} {'Total':>7} {'Subs':>6} {'Dels':>6} {'Error%':>8}")
print("-" * 40)
left_chars = sorted(
    [(ch, char_total[ch]) for ch in char_total if get_hand(ch) == "left" and char_total[ch] >= 10],
    key=lambda x: -((sum(c for (g,p),c in substitutions.items() if g==x[0]) + deletions[x[0]]) / x[1])
)
for ch, total in left_chars:
    s = sum(c for (g,p),c in substitutions.items() if g==ch)
    d = deletions[ch]
    err_rate = (s + d) / total * 100
    print(f"  {display_char(ch):<4} {total:>7} {s:>6} {d:>6} {err_rate:>7.1f}%")

print(f"\n=== Per-Character Error Rates (Right Hand) ===")
print(f"{'Char':<6} {'Total':>7} {'Subs':>6} {'Dels':>6} {'Error%':>8}")
print("-" * 40)
right_chars = sorted(
    [(ch, char_total[ch]) for ch in char_total if get_hand(ch) == "right" and char_total[ch] >= 10],
    key=lambda x: -((sum(c for (g,p),c in substitutions.items() if g==x[0]) + deletions[x[0]]) / x[1])
)
for ch, total in right_chars:
    s = sum(c for (g,p),c in substitutions.items() if g==ch)
    d = deletions[ch]
    err_rate = (s + d) / total * 100
    print(f"  {display_char(ch):<4} {total:>7} {s:>6} {d:>6} {err_rate:>7.1f}%")

# ── 3. Bar chart: left vs right per-char error rates ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, hand, chars_list, color in [
    (axes[0], "Left Hand", left_chars, "#4C72B0"),
    (axes[1], "Right Hand", right_chars, "#DD8452"),
]:
    chars_sorted = sorted(chars_list, key=lambda x: x[0].lower())
    labels_bar = [display_char(ch) for ch, _ in chars_sorted]
    rates = []
    for ch, total in chars_sorted:
        s = sum(c for (g,p),c in substitutions.items() if g==ch)
        d = deletions[ch]
        rates.append((s + d) / total * 100)

    ax.bar(range(len(labels_bar)), rates, color=color, alpha=0.8)
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels(labels_bar, fontsize=10)
    ax.set_ylabel("Error Rate (%)", fontsize=11)
    ax.set_title(f"{hand} — Per-Character Error Rate", fontsize=12)
    if rates:
        ax.axhline(y=np.mean(rates), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(rates):.1f}%")
        ax.legend()

plt.suptitle(f"Left vs Right Hand Error Rates ({args.decoder} decoding)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{args.output_prefix}_hands_{args.decoder}.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {args.output_prefix}_hands_{args.decoder}.png")

# ── Summary stats ──
print(f"\n=== Summary ===")
left_total = hand_totals["left"]
right_total = hand_totals["right"]
left_err = hand_errors["left"]
right_err = hand_errors["right"]
print(f"Left hand:  {left_err}/{left_total} errors = {left_err/left_total*100:.2f}% error rate")
print(f"Right hand: {right_err}/{right_total} errors = {right_err/right_total*100:.2f}% error rate")
if left_total > 0 and right_total > 0:
    ratio = (left_err/left_total) / (right_err/right_total)
    print(f"Ratio (left/right): {ratio:.2f}x")
