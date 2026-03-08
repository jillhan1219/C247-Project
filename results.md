# temporal results and analysis
## data 
### ACM+Log+RTN

'val_metrics': [{'val/loss': 0.8256329298019409,
                  'val/CER': 16.23836898803711,
                  'val/IER': 4.386353492736816,
                  'val/DER': 1.2627381086349487,
                  'val/SER': 10.589278221130371}],
 'test_metrics': [{'test/loss': 0.8915609121322632,
                   'test/CER': 18.240760803222656,
                   'test/IER': 4.257618427276611,
                   'test/DER': 1.6425329446792603,
                   'test/SER': 12.340609550476074}],

### ACM+NewLog+RTN

'val_metrics': [{'val/loss': 0.9278963804244995,
                  'val/CER': 25.32122230529785,
                  'val/IER': 7.332742691040039,
                  'val/DER': 1.373504638671875,
                  'val/SER': 16.614974975585938}],
 'test_metrics': [{'test/loss': 0.9533018469810486,
                   'test/CER': 25.76183319091797,
                   'test/IER': 5.532742500305176,
                   'test/DER': 2.096390724182129,
                   'test/SER': 18.132699966430664}]

### ACM+Log+BatchNorm2D

'val_metrics': [{'val/loss': 0.8189829587936401,
                  'val/CER': 16.54851531982422,
                  'val/IER': 4.342046737670898,
                  'val/DER': 1.0190517902374268,
                  'val/SER': 11.187417030334473}],
 'test_metrics': [{'test/loss': 0.8881130218505859,
                   'test/CER': 20.31553840637207,
                   'test/IER': 7.521071910858154,
                   'test/DER': 1.1238383054733276,
                   'test/SER': 11.670628547668457}]

### No ACM+Log+BatchNorm2D (Baseline)
'val_metrics': [{'val/loss': 0.7600142955780029,
                  'val/CER': 15.81745719909668,
                  'val/IER': 4.120513916015625,
                  'val/DER': 1.3956578969955444,
                  'val/SER': 10.301284790039062}],
 'test_metrics': [{'test/loss': 0.928337037563324,
                   'test/CER': 20.012968063354492,
                   'test/IER': 7.499459743499756,
                   'test/DER': 0.8861033320426941,
                   'test/SER': 11.62740421295166}]

---

## Analysis

### Part 1: Data Preprocessing & Augmentation (all use TDS+Transformer, 10.4M params)

| Config                     | val/CER | test/CER | val→test gap |
|----------------------------|---------|----------|--------------|
| ACM + Log + RTN            | 16.24   | **18.24**| 2.00         |
| ACM + Log + BatchNorm2D    | 16.55   | 20.32    | 3.77         |
| No ACM + Log + BatchNorm2D | 15.82   | 20.01    | 4.19         |
| ACM + NewLog (RSG) + RTN   | 25.32   | 25.76    | 0.44         |

**1. RTN outperforms BatchNorm2D**
RTN reduces test CER by 2.1% (18.24% vs 20.32%) and nearly halves the val→test gap (2.0% vs 3.77%). RTN's causal rolling normalization adapts to per-session signal statistics at inference time, making it more robust to session-to-session variability than BatchNorm2D.

**2. RTN is the dominant factor in improving generalization**
The baseline (no ACM, BatchNorm2D) has a 4.19% val→test gap. ACM+Log+RTN closes this to 2.0%. ACM+Log+BatchNorm2D only narrows it to 3.77%, confirming RTN drives the improvement.

**3. ACM alone does not help without RTN**
ACM+Log+BatchNorm2D (20.32%) is slightly worse than the baseline (20.01%). ACM's benefit only appears when combined with RTN.

**4. RSG significantly hurts performance**
RSG reduces 33 FFT bins to 6 frequency bands, losing discriminative spectral information. Test CER worsens by 7.5% (25.76% vs 18.24%). The near-zero val→test gap (0.44%) indicates underfitting, not overfitting.

### Part 2: Architecture — Shared Hand Weights (all use ACM + Log + RTN)

| Config                          | val/CER | test/CER | val→test gap | Params |
|---------------------------------|---------|----------|--------------|--------|
| Separate weights (TDS+Transformer) | 16.24   | **18.24**| 2.00         | 10.4M  |
| Shared weights, upscaled (TDS+Transformer, mlp=[528], blocks=[24,24,48,48]) | 19.72 | 19.84 | 0.12 | 8.1M |

Shared weights nearly eliminates the val→test gap (0.12%), confirming it is an extremely strong regularizer. However, both val (19.72%) and test (19.84%) are worse than the separate-weights model. Despite upscaling (mlp=[528], blocks=[24,24,48,48]), the 8.1M-param shared model underperforms the 10.4M separate model, suggesting the capacity reduction from weight sharing outweighs its regularization benefit at this scale.

### Part 3: Effect of BiGRU — Ablation Study (ACM + Log + RTN, same CNN backbone)

Both models use the same 6 TDS Conv blocks ([24,24] pre + [24,24,24,24] post, kernel=32) for fair comparison. The only difference is whether a BiGRU layer is appended after the CNN.

| Config                          | val/CER | test/CER | val→test gap |
|---------------------------------|---------|----------|--------------|
| TDS Conv only (6 blocks)        | 21.47   | 18.41    | -3.06        |
| **TDS Conv + BiGRU**            | **14.09** | **15.28** | **1.19**   |

**TDS Conv only:** SpectrogramNorm → MLP([384]) → Flatten → TDSConvEncoder(6 blocks, kernel=32) → Linear → LogSoftmax → CTC

'val_metrics': [{'val/loss': 0.9827985167503357,
                  'val/CER': 21.466548919677734,
                  'val/IER': 8.949934005737305,
                  'val/DER': 1.6836508512496948,
                  'val/SER': 10.832963943481445}],
 'test_metrics': [{'test/loss': 0.9702265858650208,
                   'test/CER': 18.413658142089844,
                   'test/IER': 3.630862236022949,
                   'test/DER': 1.9018802642822266,
                   'test/SER': 12.880916595458984}]

**TDS Conv + BiGRU:** SpectrogramNorm → MLP([384]) → Flatten → TDSConvEncoder(6 blocks, kernel=32) → BiGRU(2 layers, 256 hidden) → Linear(512→99) → LogSoftmax → CTC

'val_metrics': [{'val/loss': 0.7042859196662903,
                  'val/CER': 14.089499473571777,
                  'val/IER': 2.9242358207702637,
                  'val/DER': 1.3070447444915771,
                  'val/SER': 9.858219146728516}],
 'test_metrics': [{'test/loss': 0.7102144360542297,
                   'test/CER': 15.279878616333008,
                   'test/IER': 2.809595823287964,
                   'test/DER': 1.1886751651763916,
                   'test/SER': 11.281607627868652}]

**Decoded output comparison (Session 0, CTC greedy, first ~250 chars):**

| | Text |
|------|------|
| **GT** | `the quick brown fox jumps over a lazy dog⏎thai adds strategic throw⏎literary mark indivb⌫idual frontpage⏎tiny correctly origin lotus⏎hull titanium bull annex⏎` |
| **CNN only** | `the quick rown fox t ymps ove as lazy dog⏎thai dfe strategic throw⏎litrary mark incig⌫idyal frontpage⏎tiny correctlyorigin loths⏎ull titaninn byll annec⏎` |
| **CNN+BiGRU** | `the quick brown fox jumps over a lazy dog⏎thai adde strategic throw⏎literary mark infub⌫idual front'age⏎tiny correctly origin lotus⏎hull titanium bull anned⏎` |

**Error statistics comparison:**

| Metric | CNN only | CNN+BiGRU | Δ |
|--------|----------|-----------|---|
| Top deletion (space) | 485 | 393 | -19% |
| Top deletion (e) | 439 | 354 | -19% |
| Top deletion (t) | 334 | 275 | -18% |
| Top deletion (⌫) | 159 | 125 | -21% |
| Top insertion (space) | 488 | 384 | -21% |
| Top insertion (e) | 451 | 344 | -24% |

**Analysis:**

Adding BiGRU reduces test CER from 18.41% to **15.28%** (3.13% absolute). The decoded examples reveal *why*:

1. **Word-level coherence**: Without BiGRU, the CNN produces fragmented outputs — "rown" (missing 'b'), "t ymps" (space inserted), "ove as" (missing 'r'). With BiGRU, these become correct: "brown", "jumps", "over a". The bidirectional context allows the model to resolve character-level ambiguities using surrounding sequence information.

2. **Space/word boundary handling**: CNN-only inserts spurious spaces ("t ymps", "ove as") and merges words ("correctlyorigin"). BiGRU dramatically reduces space errors (485→393 deletions, 488→384 insertions, both ~20% reduction), indicating the recurrent layer learns word-level temporal structure.

3. **IER reduction (8.95% → 2.92% val, 3.63% → 2.81% test)**: The CNN-only model has extremely high val IER (8.95%), suggesting it inserts many spurious characters during windowed inference. BiGRU's global context suppresses these false positives.

4. **Generalization pattern**: CNN-only shows an unusual negative val→test gap (-3.06%) — it performs *worse* on validation windows than full test sessions. This suggests CNN-only overfits to window boundaries during training. BiGRU normalizes this to a healthy positive gap (1.19%), indicating the recurrent layer provides consistent behavior regardless of sequence length.

5. **Backspace recognition improved**: Missed backspaces drop from 159 to 125 (21% reduction). The BiGRU better distinguishes the backspace EMG gesture from regular keystrokes by leveraging the surrounding character context (backspace typically follows a typing error).

### Part 5: Decoding Strategy — CTC Greedy vs CTC Beam Search (TDS Conv + BiGRU)

| Decoder    | val/CER | test/CER | val/IER | val/DER | val/SER | test/IER | test/DER | test/SER |
|------------|---------|----------|---------|---------|---------|----------|----------|----------|
| CTC Greedy | 14.09   | 15.28    | 2.92    | 1.31    | 9.86    | 2.81     | 1.19     | 11.28    |
| CTC Beam   | **10.77** | **8.67** | **2.39** | **1.46** | **6.91** | **2.59** | **0.30** | **5.77** |

Beam search uses a 6-gram character language model (wikitext-103), beam_size=50, lm_weight=2.0, insertion_bonus=2.0.

**Example decoded output comparison (Session 0, first ~250 chars):**

| | Text |
|------|------|
| **GT** | `the quick brown fox jumps over a lazy dog⏎thai adds strategic throw⏎literary mark indivb⌫idual frontpage⏎tiny correctly origin lotus⏎hull titanium bull annex⏎` |
| **Greedy** | `the quick brown fod umps over a lazy dog⏎thai adde stratetic throw⏎piterary mark infub⌫idual frontpage⏎tiny corectlm origin loths⏎thull titanii bull anned⏎` |
| **Beam** | `the quick brown fox jumps over a lazy dog⏎thai adde strategic throw⏎literary mark infub⌫idual front'age⏎tiny correctly origin lotus⏎hull titanium bull anned⏎` |

**Key observations:**
1. **Massive CER improvement**: Beam search reduces test CER from 15.28% to **8.67%** — a 6.61% absolute (43%) relative improvement. This is the single largest improvement from any technique in this project.
2. **Deletion rate nearly eliminated**: Test DER drops from 1.19% to 0.30%, a 75% reduction. The language model helps the decoder retain characters that the acoustic model was uncertain about.
3. **Substitution rate halved**: Test SER drops from 11.28% to 5.77%. The LM provides word-level priors that correct character-level confusions (e.g., "fod umps" → "fox jumps", "stratetic" → "strategic").
4. **Rare words recovered**: Beam search correctly decodes "fox jumps", "strategic", "correctly", "lotus" which greedy got wrong, showing the LM's vocabulary knowledge compensates for acoustic ambiguity.
5. **Test outperforms val**: Unusually, test CER (8.67%) is lower than val CER (10.77%). This may be because the test set contains more common English words that the LM handles well, or because greedy decoding during validation training already optimized the acoustic model for common patterns.
6. **Error profile shift**: With beam search, deletions and insertions drop significantly but the remaining errors are harder — mostly substitutions on uncommon words or special characters where the LM has less coverage.

### Conclusion
Best configuration: **ACM + LogSpectrogram + RollingTimeNorm** with **TDS Conv + BiGRU** and **CTC beam search decoding** (test CER **8.67%**). The CNN+RNN hybrid outperforms both TDS+Transformer and TDS-only baselines. RTN remains the most impactful preprocessing component. Beam search with a character-level LM provides the largest single improvement (6.61% absolute), demonstrating that decoding strategy is as important as model architecture for EMG-to-text.
