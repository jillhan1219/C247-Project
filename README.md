# CNN + GRU Model Modification for EMG-to-Text

## Overview

This repository includes my modification of the baseline EMG-to-text model using a **CNN + GRU architecture**.

I experimented with two different training strategies:

1. **Two-stage training (no augmentation → fine-tune with augmentation)**
2. **Training from scratch with augmentation**

Character Error Rate (CER) is used as the primary evaluation metric.

---

# Model Architecture

* Backbone: **CNN + GRU**
* Input preprocessing:

  * `to_tensor`
  * `log_spec`
* Decoder: CTC-based sequence prediction

The CNN extracts local spectral patterns, while the GRU models temporal dependencies across time.

---

# Training Strategies

---

## 1️⃣ Two-Stage Training (Best Overall Stability)

### Stage 1 — Base Model (No Augmentation)

* Epochs: **40**
* Augmentation: ❌ None
  (Only `to_tensor` + `log_spec`)
* Validation CER: **15.5**
* Test CER: **17.8**

### Run Base Training

```bash
python -m emg2qwerty.train \
  user="single_user" \
  transforms=log_spectrogram_noaug \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=40
```

This stage trains a stable representation before introducing data perturbations.

---

### Stage 2 — Fine-Tuning with Gain + SoftSpec

* Additional epochs: **12**
* Dataset: Entire training set
* Augmentation:

  * `gain`
  * `soft_spec`
* Learning rate reduced for fine-tuning

#### Soft Spec Settings

* `n_time_masks: 2`
* `time_mask_param: 10`
* `n_freq_masks: 1`
* `freq_mask_param: 4`

### Run Fine-Tuning

```bash
ckpt_path={your_base_model_path}

python -m emg2qwerty.train \
  user=single_user \
  checkpoint=$ckpt_path \
  transforms=gain_softspec \
  optimizer.lr=1e-4 \
  lr_scheduler.scheduler.warmup_epochs=2 \
  num_workers=2 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=12
```

### Final Performance

* Final checkpoint Test CER ≈ **17.203**

Fine-tuning with light augmentation improves robustness and slightly reduces test CER compared to the base model.

---

## 2️⃣ Gain / SoftSpec Training From Scratch

In this experiment, the model was trained from scratch using augmentation throughout training.

### Training Setup

* Augmentation:

  * `gain`
  * `soft_spec`
* No channel masking
* Learning rate: `1e-3`
* Warmup epochs: `10`
* Total epochs: `40`

### Run From Scratch Training

```bash
HYDRA_FULL_ERROR=1 python -m emg2qwerty.train \
  user=single_user \
  transforms=gain_softspec \
  checkpoint=null \
  optimizer.lr=1e-3 \
  lr_scheduler.scheduler.warmup_epochs=10 \
  num_workers=2 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=40
```

### Results

* **Best checkpoint (Val / Test CER)**
  ≈ **17.722 / 17.0801**

(Results compared across different numbers of input channels.)

This shows that training with augmentation from the beginning can achieve competitive performance, though convergence is slightly less stable than the staged approach.

---

# Summary of Results

| Training Strategy | Epochs | Augmentation    | Test CER |
| ----------------- | ------ | --------------- | -------- |
| Base (No Aug)     | 40     | None            | 17.8     |
| Fine-tuned        | +12    | Gain + SoftSpec | ~17.203  |
| From Scratch      | 40     | Gain + SoftSpec | ~17.0801 |

---

# Setup Instructions

1. Place the modified model code into:

   ```
   emg2qwerty/
   ```

2. Place the transform configuration file into:

   ```
   config/transform/
   ```

3. Run one of the training strategies above.

---

# Future Work

* Try CNN+LSTM as backbone architecture
* Train 150 epochs for the best setting to achieve better performance

---

Author: Jenny Ho
UCLA MEng in Artificial Intelligence
