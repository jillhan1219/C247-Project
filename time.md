# AGENTS.md guidelines for codebase analysis and updates

**Session ID:** ses_3673cfb64ffezWnJSQi6HAntF4
**Created:** 2/26/2026, 7:04:03 AM
**Updated:** 2/26/2026, 7:48:28 AM

---

## User

Please analyze this codebase and create an AGENTS.md file containing:
1. Build/lint/test commands - especially for running a single test
2. Code style guidelines including imports, formatting, types, naming conventions, error handling, etc.

The file you create will be given to agentic coding agents (such as yourself) that operate in this repository. Make it about 150 lines long.
If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include them.

If there's already an AGENTS.md, improve it if it's located in /home/ubuntu/jinru/emg2qwerty

---

## Assistant

**Tool: glob**

**Tool: glob**

**Tool: bash**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: glob**

**Tool: glob**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: glob**

---

## Assistant

**Tool: read**

**Tool: glob**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: write**

---

## Assistant

**Tool: bash**

---

## Assistant

Created `/home/ubuntu/jinru/emg2qwerty/AGENTS.md` (177 lines) with:
- Build/lint/test commands (including running single tests like `pytest emg2qwerty/tests/decoder_test.py::test_logsumexp`)
- Code style guidelines for imports, formatting, type hints, naming conventions, dataclasses, error handling, testing, code headers, and pre-commit hooks
- Project overview and file structure

---

## User

In this project I need to explore different data preprocessing methods and data aumentation methods, also different model architecture, now tell me where is data preprocessing and augmentation part, and explain the code biefly to me

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

Here's where data preprocessing and augmentation happens in this project:

## Data Preprocessing & Augmentation

### 1. **Transforms** (`emg2qwerty/transforms.py`)
This is the main file for preprocessing and augmentation:

| Transform | Description |
|-----------|-------------|
| **`ToTensor`** | Converts numpy structured arrays (EMG left/right) to PyTorch tensors |
| **`LogSpectrogram`** | Converts EMG signal to log10-scaled spectrogram using STFT |
| **`SpecAugment`** | Time/frequency masking (from speech recognition paper) |
| **`RandomBandRotation`** | Augmentation: randomly shifts electrode channels |
| **`TemporalAlignmentJitter`** | Augmentation: jitters left/right EMG alignment |
| **`Compose`** | Chains multiple transforms together |
| **`ForEach`** | Applies transform to each batch item |

### 2. **Model Architectures** (`emg2qwerty/modules.py`)

| Module | Description |
|--------|-------------|
| **`SpectrogramNorm`** | Batch normalization over spectrograms |
| **`RotationInvariantMLP`** | MLP with rotation-invariant pooling over electrode offsets |
| **`MultiBandRotationInvariantMLP`** | Handles left/right bands separately |
| **`TDSConvEncoder`** | Time-Depth Separable convolutional encoder (from speech recognition) |

### 3. **Data Loading** (`emg2qwerty/data.py`)
- **`EMGSessionData`**: Reads HDF5 files containing EMG timeseries (left/right wrists, 16 channels each at 2kHz)
- **`WindowedEMGDataset`**: Creates sliding windows over EMG data with optional temporal jittering
- **`LabelData`**: Handles ground-truth text/labels from keystrokes or prompts

### 4. **Training** (`emg2qwerty/lightning.py`)
- **`WindowedEMGDataModule`**: PyTorch Lightning datamodule that applies transforms separately for train/val/test sets

### Key Points
- Training uses `jitter=True` for temporal alignment jitter augmentation; validation/test use `jitter=False`
- Config files in `config/` and `hydra_configs/` define which transforms to use
- Override configs via CLI: `python -m emg2qwerty.train key1=value1`

---

## User

/export data.txt

---

## Assistant

**Tool: write**

---

## Assistant

Written to `/home/ubuntu/jinru/emg2qwerty/data.txt`. The LSP errors are due to missing dependencies in the environment - not actual code issues.

---

## User

First let's focus on data part. Tell me what preprocessing and augmentation have done in the baseline

---

## Assistant

**Tool: glob**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

Based on the config files, here's what the **baseline** does:

## Baseline Preprocessing & Augmentation

### Preprocessing (Train/Val/Test)
| Step | Transform | Description |
|------|-----------|-------------|
| 1 | **`ToTensor`** | Convert HDF5 EMG data (left/right, 16 channels each) to PyTorch tensor |
| 2 | **`LogSpectrogram`** | Convert EMG to log10-scaled spectrogram (n_fft=64, hop_length=16 → 2kHz→125Hz) |

### Augmentation (Training Only)
| Transform | Description |
|-----------|-------------|
| **`RandomBandRotation`** | Randomly shift electrode channels by -1, 0, or +1 |
| **`TemporalAlignmentJitter`** | Randomly jitter left/right alignment by ±60ms (±120 timesteps at 2kHz) |
| **`SpecAugment`** | Time masking (3 masks, max 200ms) + Frequency masking (2 masks, max 4 bins) |

### Pipeline Order
```
Train: ToTensor → BandRotation → TemporalJitter → LogSpectrogram → SpecAugment
Val:   ToTensor → LogSpectrogram
Test:  ToTensor → LogSpectrogram
```

### Key Points
- Augmentations are **only applied during training** (val/test skip them)
- `TemporalAlignmentJitter` is applied via `WindowedEMGDataset(jitter=True)` in lightning.py, not as a transform
- The baseline uses TDSConvCTC model from `config/model/tds_conv_ctc.yaml`

---

## User

Read the code in @SplashNet, it is another work on the same kind of data, but it uses more data than I do, focus on the data preprocessing part and augmentation part, find out what methods does it use to improve?

---

## Assistant

**Tool: glob**

**Tool: glob**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

Could you provide more context about SplashNet? Is it:
1. A paper title (arXiv ID or link)?
2. A GitHub repository URL?
3. Located in a specific directory?

Let me also check if there's a related work section in this codebase:

**Tool: read**

---

## Assistant

**Tool: webfetch**

---

## Assistant

**Tool: webfetch**

---

## Assistant

Could you provide more context about SplashNet? For example:
- A link to the paper or GitHub repository?
- Is it related to the same EMG typing task?

From my search, I couldn't find a project called "SplashNet" specifically for EMG typing. The main related work I found is:
- **emg2qwerty** (Meta's NeurIPS 2024 paper) - the baseline for this codebase

If you can share a URL or paper title, I can look up the data preprocessing and augmentation methods used in that work.

---

## User

It is a file under my dir, @SplashNet/ 

---

## Assistant

**Tool: glob**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: glob**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

Now I understand SplashNet's approach. Here's the comparison:

## SplashNet vs Baseline - Data Preprocessing & Augmentation

### Shared Methods (same as baseline)
| Method | Description |
|--------|-------------|
| **`ToTensor`** | Convert EMG to PyTorch tensor |
| **`RandomBandRotation`** | Random electrode channel shift (-1, 0, +1) |
| **`TemporalAlignmentJitter`** | ±60ms jitter on left/right alignment |
| **`SpecAugment`** | Time/frequency masking |

### SplashNet's Additional/Modified Methods

| Method | Description | Benefit |
|--------|-------------|---------|
| **`NewLogSpectrogram`** (RSG) | Aggregates 33 frequency bins → 6 bands (based on EMG frequency ranges) | Reduces spectral granularity, speeds up training |
| **`RollingTimeNorm`** (RTN) | Replaces BatchNorm2d with rolling time normalization | Better for streaming/longer sequences |
| **`Longer Training Clips`** | 16s windows (vs 4s baseline) | More context for better decoding |
| **ACM** (Aggressive Channel Masking) | freq_mask_param=12 (vs 4), n_time_masks=0 | More aggressive frequency masking |

### Key Config Changes in SplashNet

```yaml
# 1. Longer clips
window_length: 32000  # 16s instead of 8000 (4s)

# 2. Reduced Spectral Granularity (RSG)
logspec:
  _target_: emg2qwerty.transforms.NewLogSpectrogram  # 6 frequency bands

# 3. Rolling Time Normalization (RTN)
spec_norm: 'RollingTimeNorm'

# 4. Aggressive Channel Masking (ACM)
freq_mask_param: 12  # was 4
n_time_masks: 0     # disable time masking
```

### Summary
SplashNet improves upon baseline by:
1. **Longer context** (16s vs 4s windows)
2. **Reduced frequency bins** (6 bands vs 33) - speeds up computation
3. **Rolling normalization** - better for real-time streaming
4. **More aggressive augmentation** - freq masking with larger masks

---

