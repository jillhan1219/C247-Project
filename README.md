# Deep Learning for Keystroke Prediction from sEMG Signals

ECE C147/C247 Final Project — Winter 2026

## Team
- Yudi Chen
- Jenny Ho
- Jiwon Bae
- Jinru Han

## Overview

This project decodes QWERTY keystrokes from wrist-mounted surface EMG (sEMG) signals using hybrid convolutional-recurrent neural networks. Our best model (TDS Conv + BiGRU with CTC loss) achieves **8.67% CER** with beam search decoding on a single subject.

Built upon [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) from Meta.

## Setup

```bash
# Create conda environment
conda create -n scaling python=3.10
conda activate scaling
pip install -e .

# Download data (place under data/)
# See original emg2qwerty repo for data download instructions
```

## Training

```bash
# Train the best model (TDS Conv + BiGRU)
conda run -n scaling python -m emg2qwerty.train user=single_user model=bigru_ctc

# Train the baseline (TDS Conv only)
conda run -n scaling python -m emg2qwerty.train user=single_user model=tds_conv_ctc
```

Key training flags:
- `model=bigru_ctc` — TDS Conv + BiGRU (best)
- `model=tds_conv_ctc` — TDS Conv baseline

## Testing

```bash
# Greedy decoding
conda run -n scaling python -m emg2qwerty.train user=single_user train=False \
    checkpoint=<path_to_ckpt> decoder=ctc_greedy

# Beam search with language model
conda run -n scaling python -m emg2qwerty.train user=single_user train=False \
    checkpoint=<path_to_ckpt> decoder=ctc_beam --multirun
```

## Project Structure

### Main Pipeline (Jinru)

| Path | Description |
|------|-------------|
| `emg2qwerty/modules.py` | Model architectures (TDS Conv, BiGRU, Transformer, MLP, norms) |
| `emg2qwerty/lightning.py` | PyTorch Lightning modules (`TDSConvCTCModule`, `BiGRUCTCModule`) |
| `emg2qwerty/transforms.py` | Data preprocessing and augmentation (LogSpectrogram, ACM, RTN) |
| `emg2qwerty/train.py` | Training entry point |
| `emg2qwerty/decoder.py` | CTC greedy and beam search decoders |
| `config/model/` | Model configs (`bigru_ctc.yaml`, `tds_conv_ctc.yaml`) |
| `config/transforms/` | Transform and augmentation configs |
| `config/base.yaml` | Base training config |

### Teammates' Architecture Explorations

| Folder | Author | Description |
|--------|--------|-------------|
| `jenny_cnn_rnn_exp/` | Jenny Ho | CNN + RNN hybrid experiments |
| `jiwon-rnn-exp/` | Jiwon Bae | Standalone RNN (BiGRU, BiLSTM) experiments |
| `yudi_cnn_exp/` | Yudi Chen | CNN-only architecture (TCN) experiments |

## Best Model Configuration

- **Architecture**: TDS Conv (4 blocks, ch=24, kernel=32) + BiGRU (2 layers, hidden=256)
- **Normalization**: RollingTimeNorm (RTN)
- **Augmentation**: Aggressive Channel Masking (freq_mask_param=12, no time masks)
- **Spectrogram**: LogSpectrogram (n_fft=64, hop=16, 33 freq bins)
- **Decoder**: CTC beam search with 6-gram character LM (beam=50, lm_weight=2.0)
- **Test CER**: 15.28% (greedy) / 8.67% (beam search)
