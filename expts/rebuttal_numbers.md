## M1. RT pretrained on PluRel data with the same recipe

RT-P vs RT-J task-wise results below.

## M2.1. Attribution of gains: corpus scale

Data scaling results below.

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
