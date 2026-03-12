# ECE C147/C247 Winter 2026 Final Project: How To Max Your Score

Due date: **Friday, March 13, 2026 (Week 10 Friday)**.

This guide rewrites `26W_ECE147_247_FinalProject.pdf` as a point-focused checklist for scoring high.

## What You Must Do (Non-Negotiables)

- Use **post-CNN** sequence modeling (examples listed: **RNNs, CNN+RNN, Transformers**).
- For the baseline emg2qwerty project, you **must** experiment with **at least 1 recurrent architecture** (RNN/LSTM/GRU).
- Use the provided **train/val/test split** for subject **#89335547** from `single_user.yaml` (linked in the PDF).
- Evaluate using **Character Error Rate (CER)**.
- Submit a write-up (baseline: **<= 7 pages**, custom: **<= 9 pages**, references excluded).
- Submit your **code** (so staff can validate results).
- Write-up format: **NeurIPS 2024 template**; remove line numbers via `\\usepackage[final]{neurips 2024}`.

## “Baseline” Project Setup (What The PDF Assumes)

- Task: predict QWERTY keystrokes from wrist sEMG (`B=2` bands, `C=16` electrodes each, `32` channels total, `2kHz` sampling).
- Ground truth is an unsegmented character sequence, so the PDF recommends using **CTC loss** (`CTCLoss`).
- Starter implementation: `https://github.com/Calvin-Pang/emg2qwerty`.
- Reported baseline training: ~**1 hour** on Colab for **40 epochs**, reaching roughly **30 validation CER** (for one subject).
- You can earn **full credit** using only subject **#89335547**.

## Grading Rubric (How Points Are Actually Awarded)

The project is graded out of **20 points**, but rubric categories sum to **> 20** and your score is **capped at 20**. That means you want to be strong in multiple categories and “overachieve” in at least one.

- Creativity: **7 points**
- Insight: **7 points**
- Performance: **6 points**
- Write-up: **4 points**

## How To Score High (Mapped To The Rubric)

### Creativity (7): maximize architectural diversity and experimental scope

Do enough that your project clearly goes beyond “I trained one model”.

- Compare **multiple architecture families**, not just variants.
- Include at least one RNN-family model (required): LSTM/GRU (optionally bidirectional).
- Add at least one non-RNN sequential model: Transformer-style encoder, temporal convolution stack, CNN+RNN hybrid, etc.
- Include the repository baseline (TDS) or a faithful reproduction as an anchor reference point.
- Try at least 1 “project direction” beyond architectures:
- Preprocessing or augmentation study.
- Channel-count ablation (how many electrodes needed vs CER).
- Data-volume ablation (how much training data needed vs CER).
- Sampling-rate ablation (downsample vs CER).
- Make comparisons fair:
- Keep splits fixed (use the provided split).
- Use comparable training budgets, or explicitly account for differences.

Deliverable that screams “creativity”: a table like “Model family vs CER vs params vs training time” plus 1 ablation figure.

### Insight (7): explain *why* results happen (not just “X is better”)

You get these points by doing analysis, not by adding more models.

- Add at least 2 of these analysis components:
- Error analysis: which characters are most substituted/deleted/inserted; CER by character type (letters vs punctuation) or by frequency.
- Qualitative decoding examples: short sequences showing typical failure modes (insertions, repeats, missed keystrokes).
- Ablation-based reasoning: connect a controlled change (e.g., channel dropout, downsampling) to a hypothesized mechanism.
- Training dynamics: learning curves; overfitting discussion; effect of regularization.
- Turn observations into testable hypotheses:
- Example pattern: “Downsampling hurts fast keystrokes” -> show CER vs sampling rate and discuss temporal resolution limits.
- Example pattern: “Some channels contribute little” -> show CER vs channel count; discuss redundancy across electrodes.

### Performance (6): beat baselines *and* show you optimized reasonably

Performance is relative to limited resources; staff expects you to do more than “default settings”.

- Establish a clear baseline you can reproduce (e.g., repo baseline, or your own simple RNN).
- Improve CER with at least 2 levers:
- Architecture changes (depth, bidirectionality, attention, CNN front-end).
- Optimization changes (learning rate schedule, weight decay, dropout, gradient clipping).
- Input pipeline changes (normalization, window length, resampling).
- Report performance properly:
- CER on the relevant split(s) (val and test if you run it).
- If you ran multiple seeds, report mean/std (even 2-3 seeds is a step up).

### Write-up (4): be clear, reproducible, and easy to grade

Treat the write-up like a mini NeurIPS paper.

- Follow the required sectioning:
- Abstract, Introduction (motivated *question*, not generic sEMG), Methods, Results, Discussion, References.
- Make results scannable:
- One main table of CER numbers.
- One figure for the most important ablation (channels, data amount, or sampling rate).
- Make it reproducible for validation:
- State dataset used (subject #89335547), splits source, preprocessing steps, and training settings.
- Include enough detail that staff can run your code and match your reported numbers.

## A “Max-Score” Minimal Project Recipe (If You Want The Shortest Path)

If you do only subject #89335547, a strong high-score scope is:

1. Baseline reproduction (TDS or simple RNN) with CER reported.
2. Additional architecture family: CNN+BiGRU (or CNN+BiLSTM).
3. Additional architecture family: Transformer encoder model.
4. One ablation study: channel-count vs CER (e.g., 32, 24, 16, 8 channels).
4. One optimization study:
5. One optimization study: a small sweep over learning rate / dropout / weight decay, showing that you tuned.
6. Insight section: error analysis (top error characters or sample decoded sequences) plus hypotheses tied to your ablation.

## Submission Checklist

- Write-up:
- Uses NeurIPS 2024 style.
- <= 7 pages (baseline) or <= 9 pages (custom), references excluded.
- Contains the required sections.
- Code:
- Runs end-to-end to reproduce your reported CER.
- Uses the provided split for subject #89335547 (unless you are explicitly expanding the dataset).
- Clearly documents how to train/evaluate (README + exact commands).

## Notes On Custom Projects

The PDF says custom projects require emailing Prof. Kao for approval by **Feb 23, 2025**, including what post-CNN architectures you will use. As of **March 2026**, that date is already in the past, so assume you should follow the baseline project unless you already have prior approval.
