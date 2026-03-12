## Creativity (7 pts): “Did you try diverse approaches?”
**To-do**
- Reproduce the provided baseline (TDS from the repo) on subject `#89335547` with the official split; record val CER and test CER.
- Implement and train at least 2 additional *distinct* architecture families (3 total models minimum):
  - Model A (required recurrent): `CNN -> BiGRU (or BiLSTM) -> linear -> CTC`.
  - Model B (non-RNN): Transformer encoder (or Conformer-lite) with CTC head.
- Add 1 preprocessing/augmentation experiment and show whether it helps:
  - Examples: per-channel normalization choice, bandpass/notch, time masking, channel dropout, Gaussian noise, random time shift.
- Add 1 “project direction” ablation and plot CER vs the variable:
  - Channels ablation: 32/24/16/8 channels (choose a consistent selection rule).
  - Or sampling-rate ablation: 2000 Hz vs downsampled variants.
  - Or data-amount ablation: 100%/50%/25% of training sessions.
- Put all models and ablations into one comparison table (params, training time, best-val CER, test CER).

**Done-when**
- You have 3 models + 1 ablation plot + 1 augmentation/preprocess study, all on the official split.

---

## Insight (7 pts): “Do you explain why things worked or failed?”
**To-do**
- Do a structured error analysis (not just “CER improved”):
  - Show substitution/deletion/insertion rates (or at least qualitative examples of each).
  - Show the most common confusions (top character-pair substitutions).
- Add 6–10 decoded examples (short snippets) comparing:
  - Ground truth vs baseline vs your best model.
  - Annotate what failure mode is happening (missed keystrokes, repeated chars, timing drift).
- For your ablation (channels or sampling rate or data amount), write 2–3 concrete hypotheses and check them against the results.
- Include training diagnostics for at least 2 models:
  - Train/val CER curves (or loss + CER) to discuss under/overfitting and optimization stability.
- If your best model beats others, explain *what changed* mechanistically (temporal context, capacity, regularization, inductive bias).

**Done-when**
- Your Discussion section contains specific failure modes + hypotheses tied to your plots/tables.

---

## Performance (6 pts): “Did you optimize reasonably and get good results?”
**To-do**
- Pick one strong model (likely CNN+BiGRU or Transformer) and do a small but real tuning sweep:
  - Learning rate (3 values), dropout (2 values), weight decay (2 values) or similar (keep total runs manageable).
- Add at least 2 standard training improvements and report their effect:
  - LR schedule (cosine/one-cycle), gradient clipping, SpecAugment-style time masking, label smoothing (if applicable), better normalization.
- Compare decoding strategies (if feasible):
  - Greedy vs beam search decoding for CTC, report CER difference and runtime.
- Report results properly:
  - Best val CER per model, and corresponding test CER.
  - Keep compute budget roughly comparable across models, or explicitly state differences.

**Done-when**
- You can show “baseline -> improved model” with clear tuning steps and measurable CER gains.

---

## Write-up (4 pts): “Is it clear and easy to grade?”
**To-do**
- Follow NeurIPS 2024 format and required sections (Abstract/Intro/Methods/Results/Discussion/Refs).
- Make Results extremely scannable:
  - One main table: all models with best-val CER, test CER, params, training time.
  - One main figure: your ablation curve (channels or sampling or data amount).
- Make Methods reproducible:
  - Exact dataset (subject `#89335547`), split source, windowing, preprocessing, optimizer, schedule, batch size, epochs, early stopping criterion.
  - State how CER is computed and how decoding is done.
- Add a short “Reproducibility” paragraph:
  - How to run training/eval (commands), hardware used, and where configs live in your repo.

**Done-when**
- A TA can find your main table/figure in under 30 seconds and rerun your best model from your repo instructions.

If you tell me what codebase you’re using (the official `emg2qwerty` repo as-is vs your own training loop), I can turn this into a concrete experiment matrix (exact runs/config names) sized to your compute budget.

