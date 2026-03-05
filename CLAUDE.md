# CLAUDE.md - Project Configuration for Claude Code

## Environment
- Conda environment: `scaling`
- Run commands with: `conda run -n scaling <command>`
- Python version: 3.10

## Quick Reference
```bash
# Training
conda run -n scaling python -m emg2qwerty.train user=single_user

# Testing (with checkpoint)
conda run -n scaling python -m emg2qwerty.train user=single_user train=False checkpoint=<path> decoder=ctc_greedy --multirun

# Run tests
conda run -n scaling pytest
```

## Key Files
- `emg2qwerty/modules.py` — Model architecture (TDS Conv, MLP, SpectrogramNorm, RollingTimeNorm, TransformerEncoder)
- `emg2qwerty/lightning.py` — PyTorch Lightning module and data module
- `emg2qwerty/transforms.py` — Data preprocessing and augmentation
- `emg2qwerty/train.py` — Training entry point
- `config/model/tds_conv_ctc.yaml` — Model and datamodule config
- `config/transforms/log_spectrogram.yaml` — Transform and augmentation config
- `config/base.yaml` — Base training config (epochs, seed, callbacks)

## Current Best Config (test CER 18.24%)
- Normalization: RollingTimeNorm (RTN)
- Augmentation: Aggressive Channel Masking (ACM) — n_time_masks=0, freq_mask_param=12
- Spectrogram: LogSpectrogram (33 bins), NOT NewLogSpectrogram/RSG (6 bins)
- Test: use window_length=None with transformer_chunk_size=5000, NOT windowed testing

## Architecture Notes
- SplashNet reference implementation is in `SplashNet/` — do NOT import from it directly
- `share_hand_weights` splits features into left/right halves, processes both with shared weights
- When share_hand_weights=True: MLP uses 1 shared MLP, TDS conv reshapes (T,N,C)→(T,2N,C/2)
