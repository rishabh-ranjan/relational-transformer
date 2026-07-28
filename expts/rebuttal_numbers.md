## M1. RT pretrained on PluRel data with the same recipe

RT-P vs RT-J task-wise results below.

## M2.1. Attribution of gains: corpus scale

Data scaling results below. Best-val checkpoints of RT pretrained on 10%,
32%, and 100% (rt-j) of the pretraining corpus, evaluated on the full
RelBench test split (ctx 4096). 10% caveat: that run failed at step
44000/100001, best-val ckpts from steps 6000/8000.

**Classification (ROC-AUC %, higher is better)**

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-churn | 77.14 | **79.86** | 77.98 |
| rel-amazon/user-churn | 64.82 | **68.12** | 67.49 |
| rel-avito/user-clicks | 46.48 | **53.84** | 51.91 |
| rel-avito/user-visits | 51.96 | 54.79 | **58.64** |
| rel-event/user-ignore | 79.77 | **84.22** | 83.80 |
| rel-event/user-repeat | 69.57 | 64.93 | **75.63** |
| rel-f1/driver-dnf | 77.77 | 79.88 | **80.82** |
| rel-f1/driver-top3 | 84.92 | **89.68** | 88.18 |
| rel-hm/user-churn | 57.99 | **62.28** | 60.76 |
| rel-stack/user-badge | **79.45** | 76.70 | 76.82 |
| rel-stack/user-engagement | 85.82 | 85.32 | **87.03** |
| rel-trial/study-outcome | 58.72 | 61.05 | **62.97** |
| **mean** | 69.53 | 71.72 | **72.67** |

**Regression (normalized MAE %, lower is better)**

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-ltv | 9.70 | **8.55** | 9.19 |
| rel-amazon/user-ltv | 31.98 | **28.79** | 29.79 |
| rel-avito/ad-ctr | 46.31 | 47.46 | **43.38** |
| rel-event/user-attendance | **36.71** | 39.63 | 39.46 |
| rel-f1/driver-position | 49.25 | 43.33 | **40.46** |
| rel-hm/item-sales | 16.24 | 12.75 | **12.71** |
| rel-stack/post-votes | 17.86 | 15.23 | **14.76** |
| rel-trial/site-success | 40.15 | 33.55 | **32.31** |
| rel-trial/study-adverse | 16.65 | 17.21 | **16.28** |
| **mean** | 29.43 | 27.39 | **26.48** |

## M2.2. Relational access

TODO


## M3. RFM baselines

Griffin doesn't support ICL.
RDB-PFN doesn't support regression.
RDB-PFN classification numbers below.

## M4. Wall-clock and label budget for tuning

Argue with train/val sizes and tuning effect on rel-f1. TODO


## M6. Evaluation beyond RelBench

4 classification tasks from 3 databases.
Results below.


## R2-W3. Random walk retriever sensitivity

Timings below.

ms/item per example on 1x B200 GPU with 1x AMD EPYC 9565 CPU.
Mean $\pm$ std. dev. over 3 runs.

Stage | (W,K) = (10k, 20) | (W,K) = (10k, 10) | (W,K) = (1k, 20) | (W,K) = (1k, 10)
--- | --- | --- | --- | ---
Context sampler (CPU) | 24.84 $\pm$ 4.60 | 10.80 $\pm$ 2.20 | 6.05 $\pm$ 0.75 | 4.30 $\pm$ 0.39
Forward pass (GPU) | 26.00 $\pm$ 0.10 | 26.17 $\pm$ 0.10 | 25.99 $\pm$ 0.12 | 26.47 $\pm$ 0.17
Total | 50.84 $\pm$ 4.63 | 36.97 $\pm$ 2.17 | 32.04 $\pm$ 0.71 | 30.77 $\pm$ 0.30




Task-wise quality below.

AUROC % $\uparrow$, mean $\pm$ std. dev. over 3 runs.
Bold numbers indicate comparable with best result within 1 std. dev.

Task | (W,K) = (10k, 20) | (W,K) = (10k, 10) | (W,K) = (1k, 20) | (W,K) = (1k, 10)
--- | --- | --- | --- | ---
rel-f1/driver-dnf | **81.37 $\pm$ 0.41** | **81.48 $\pm$ 0.33** | **81.29 $\pm$ 0.41** | **81.36 $\pm$ 0.27**
rel-f1/driver-top3 | **91.03 $\pm$ 0.07** | **90.84 $\pm$ 0.34** | **90.71 $\pm$ 0.34** | 90.22 $\pm$ 0.35
rel-event/user-ignore | **83.31 $\pm$ 0.30** | **82.92 $\pm$ 0.33** | 82.80 $\pm$ 0.07 | **83.05 $\pm$ 0.50**
rel-event/user-repeat | **74.33 $\pm$ 1.74** | **73.46 $\pm$ 1.16** | **73.80 $\pm$ 0.80** | **73.39 $\pm$ 1.38**
rel-trial/study-outcome | **61.95 $\pm$ 1.52** | 56.64 $\pm$ 0.79 | 53.95 $\pm$ 0.74 | 51.62 $\pm$ 1.01
Mean | **78.40 $\pm$ 0.50** | 77.07 $\pm$ 0.25 | 76.51 $\pm$ 0.20 | 75.93 $\pm$ 0.47

nMAE % $\downarrow$, mean $\pm$ std. dev. over 3 runs.
Bold numbers indicate comparable with best result within 1 std. dev.

Task | (W,K) = (10k, 20) | (W,K) = (10k, 10) | (W,K) = (1k, 20) | (W,K) = (1k, 10)
--- | --- | --- | --- | ---
rel-f1/driver-position | **41.43 $\pm$ 0.06** | 42.08 $\pm$ 0.07 | 42.39 $\pm$ 0.11 | 42.80 $\pm$ 0.12
rel-event/user-attendance | **35.68 $\pm$ 0.09** | **35.78 $\pm$ 0.13** | **35.84 $\pm$ 0.11** | **35.87 $\pm$ 0.17**
rel-trial/site-success | 31.42 $\pm$ 0.09 | **31.27 $\pm$ 0.05** | 35.62 $\pm$ 0.01 | 38.66 $\pm$ 0.15
rel-trial/study-adverse | **15.50 $\pm$ 0.06** | 15.94 $\pm$ 0.04 | 16.27 $\pm$ 0.08 | 16.83 $\pm$ 0.02
Mean | **31.01 $\pm$ 0.03** | 31.27 $\pm$ 0.03 | 32.53 $\pm$ 0.07 | 33.54 $\pm$ 0.09


## R3-W3. Computational cost

Compute scaling results below.

% of full pretraining | Avg. AUROC % $\uparrow$ | Avg. nMAE % $\downarrow$
--- | --- | ---
10 | 69.98 | 35.95
20 | 69.17 | 32.26
30 | 69.24 | 30.83
40 | 70.51 | 30.04
50 | 71.08 | 28.96
100 | 72.18 | 27.91
