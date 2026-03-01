# EMG2QWERTY – CNN-CTC v2 + SE 

This document explains the purpose, usage, and results of the **CNN-CTC v2 + SE** experiment.
It is intended for **team members** who need a clear and quick understanding of the added files and how to run the code.

---

## File Overview

### 1. `emg2qwerty/lightning.py`

**What it is**
- A Python file containing PyTorch Lightning modules.
- Implements the model **CNNCTCModuleV2_SE**.

**What it does**
- Defines the neural network architecture and the full training / validation / testing logic.
- Uses CTC loss for alignment-free EMG-to-text sequence learning.
- Logs loss and error metrics automatically during validation and testing.

**How it is used**
- Instantiated by Hydra via:
emg2qwerty.lightning.CNNCTCModuleV2_SE

- This file is **not run directly**.
- It is called internally when running `python -m emg2qwerty.train`.

---

### 2. `config/model/cnn_ctc_v2_se.yaml`

**Where it lives**
config/model/cnn_ctc_v2_se.yaml


**What it is**
- A Hydra configuration file for the CNN-CTC v2 + SE model.

**What it controls**
- Model class selection (`CNNCTCModuleV2_SE`)
- Model hyperparameters:
  - Number of channels
  - Kernel size
  - Number of TCN blocks
  - Dilation growth
  - Dropout rate
  - SE attention parameters
- EMG windowing and padding settings

**Why it matters**
- The argument `model=cnn_ctc_v2_se` maps directly to this file.
- All architectural or hyperparameter changes should be made here, not in the code.

---

## How to Run

```bash
python -m emg2qwerty.train \
  user=single_user \
  model=cnn_ctc_v2_se \
  trainer.accelerator=gpu trainer.devices=1 \
  dataset.root=/path/to/your/data
```

## Important Notes

- `dataset.root` is **not fixed**. Each user should set it to their own dataset location.
- `model=cnn_ctc_v2_se` must **exactly match** the YAML filename in `config/model/`.

---

## Model Structure: CNN-CTC v2 + SE

- **Residual TCN Blocks**  
  Residual connections (x + y) for stable deep temporal modeling

- **Dilated 1D Convolutions**  
  Monotonic dilation growth to expand receptive field without downsampling

- **Pre-Activation (Pre-Norm)**  
  BatchNorm → GELU → Dropout → Conv for improved training stability

- **Batch Normalization (BN1d)**  
  Stabilizes feature distribution and accelerates convergence

- **Dropout**  
  Regularization inside each TCN block

- **SE Channel Attention (SE-Gating)**  
  Adaptive channel reweighting via global pooling and gating

- **Stride = 1 with Padding**  
  Preserves temporal length for strict CTC alignment

- **1×1 Conv Projections**  
  Efficient feature dimension mapping before and after TCN

- **CTC Classification Head**  
  LogSoftmax + CTCLoss for alignment-free sequence learning

- **PyTorch Lightning Utilities**  
  Structured training loop, logging, metrics, and optimizer scheduling

---

## Example Results

### Validation Metrics
- **CER**: 15.04  
- **DER**: 3.15  
- **IER**: 3.41  
- **SER**: 8.48  
- **Loss**: 0.75  

### Test Metrics
- **CER**: 16.64  
- **DER**: 2.10  
- **IER**: 3.37  
- **SER**: 11.17  
- **Loss**: 0.83  

