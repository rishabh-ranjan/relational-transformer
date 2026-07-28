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

Task-wise quality below.

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
