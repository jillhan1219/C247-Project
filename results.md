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

### Summary Table (test CER %)

| Config                     | val/CER | test/CER | val→test gap |
|----------------------------|---------|----------|--------------|
| ACM + Log + RTN            | 16.24   | **18.24**| 2.00         |
| ACM + Log + BatchNorm2D    | 16.55   | 20.32    | 3.77         |
| No ACM + Log + BatchNorm2D | 15.82   | 20.01    | 4.19         |
| ACM + NewLog (RSG) + RTN   | 25.32   | 25.76    | 0.44         |

### Key Findings

**1. RTN outperforms BatchNorm2D (ACM+Log+RTN vs ACM+Log+BatchNorm2D)**
RTN reduces test CER by 2.1% (18.24% vs 20.32%) and nearly halves the val→test gap (2.0% vs 3.77%). BatchNorm2D computes statistics over the training batch, which does not generalize as well to unseen test sessions. RTN's causal rolling normalization adapts to the signal statistics of each session at inference time, making it more robust to session-to-session variability.

**2. RTN reduces train→test generalization gap most significantly**
The baseline (no ACM, BatchNorm2D) has a 4.19% val→test gap — the largest among all configs. ACM+Log+RTN closes this gap to just 2.0%, suggesting RTN is the dominant factor in improving generalization rather than ACM alone. ACM+Log+BatchNorm2D also narrows the gap slightly vs the baseline (3.77%), confirming ACM provides some regularization benefit but is insufficient on its own.

**3. ACM alone does not help without RTN**
Comparing ACM+Log+BatchNorm2D (test CER 20.32%) vs the baseline No ACM+Log+BatchNorm2D (test CER 20.01%), ACM slightly worsens results in this setting. The heavy frequency masking (freq_mask_param=12) may be too aggressive when paired with BatchNorm2D, as the normalization already provides some implicit regularization. ACM's benefit appears only when combined with RTN.

**4. RSG (NewLog) significantly hurts performance**
ACM+NewLog+RTN achieves 25.76% test CER — 7.5% worse than ACM+Log+RTN. Compressing 33 FFT bins into 6 frequency bands loses discriminative spectral information that the model relies on to distinguish similar keystrokes. The val→test gap is negligible (0.44%), suggesting the model underfits rather than overfits — the reduced representation simply lacks the capacity to capture the full signal.

### Conclusion
The best configuration is **ACM + LogSpectrogram + RollingTimeNorm** (test CER 18.24%). RTN is the most impactful component, improving both absolute CER and generalization. RSG should not be used as it causes significant information loss with this model architecture.
