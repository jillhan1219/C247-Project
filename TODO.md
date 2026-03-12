# Project TODO — ECE C147/C247 Final Project

**Due: Friday, March 13, 2026**
**Subject: #89335547, official split from single_user.yaml**
**Score: 20 pts max (Creativity 7 + Insight 7 + Performance 6 + Write-up 4, capped at 20)**

---

## Priority 1: REQUIRED — Missing Architecture (Creativity)

- [ ] **Implement CNN → BiGRU → Linear → CTC model**
  - Required by rubric ("must experiment with at least 1 recurrent architecture")
  - Architecture: SpectrogramNorm → MLP → Flatten → BiGRU (2 layers) → Linear → LogSoftmax → CTC
  - Train with same ACM+Log+RTN preprocessing
  - Record: params, training time, val CER, test CER

## Priority 2: HIGH — Ablation Study with Plot (Creativity + Insight)

- [ ] **Data-amount ablation: 100% / 50% / 25% training sessions**
  - Create 2 additional user configs with fewer training sessions
  - Train best model (TDS+Transformer, ACM+Log+RTN) on each subset
  - Plot: CER vs % training data
  - Write 2-3 hypotheses (e.g., "CER degrades sublinearly because EMG patterns are repetitive across sessions")

## Priority 3: HIGH — Error Analysis & Decoded Examples (Insight)

- [ ] **Write inference script that outputs decoded examples**
  - Run best checkpoint, collect 6-10 short decoded snippets
  - Show side-by-side: ground truth vs baseline vs best model
  - Annotate failure modes: missed keystrokes, repeated chars, timing drift
- [ ] **Confusion analysis**
  - Extract top-10 character-pair substitutions from test set
  - Show IER/DER/SER breakdown (already have this in results)

## Priority 4: MEDIUM — Training Curves (Insight)

- [ ] **Extract train/val CER curves from TensorBoard logs**
  - Need curves for at least 2 models (e.g., baseline TDS vs TDS+Transformer)
  - Plot and discuss overfitting/convergence behavior
  - Compare how RTN vs BatchNorm affects training dynamics

## Priority 5: MEDIUM — Beam Search Comparison (Performance)

- [ ] **Test best checkpoint with decoder=ctc_beam vs decoder=ctc_greedy**
  - Already have ctc_beam config in config/decoder/
  - Report CER difference and runtime difference
  - Easy points for Performance section

## Priority 6: LOW — Hyperparameter Sweep (Performance)

- [ ] **Small sweep on best model**
  - Try 3 learning rates (e.g., 0.0005, 0.001, 0.002)
  - Try 2 dropout values if applicable
  - Show "baseline → tuned" improvement path

## Priority 7: Write-up (Write-up 4 pts)

- [ ] **NeurIPS 2024 format paper (≤7 pages + refs)**
  - Sections: Abstract, Introduction, Methods, Results, Discussion, References
  - Main table: all models (TDS baseline, TDS+Transformer, BiGRU) with val CER, test CER, params, training time
  - Main figure: ablation plot (data amount or channels vs CER)
  - Reproducibility paragraph: commands, hardware, config locations
  - Use `\usepackage[final]{neurips_2024}`

---

## What's Already Done ✅

- [x] Baseline TDS model reproduced
- [x] Transformer encoder model (TDS+Transformer)
- [x] Preprocessing study: ACM vs standard SpecAugment
- [x] Normalization study: RTN vs BatchNorm2D
- [x] RSG (Reduced Spectral Granularity) experiment
- [x] Shared hand weights experiment
- [x] Results table with analysis in results.md

## Models for Main Comparison Table

| Model | Status | val CER | test CER | Params |
|-------|--------|---------|----------|--------|
| TDS baseline (BatchNorm, no ACM) | ✅ Done | 15.82 | 20.01 | 10.4M |
| TDS+Transformer (ACM+RTN) | ✅ Done | 16.24 | 18.24 | 10.4M |
| CNN → BiGRU → CTC | ❌ TODO | — | — | — |
