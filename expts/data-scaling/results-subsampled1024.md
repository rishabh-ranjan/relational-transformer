# Data-scaling eval: test split, items_per_task=1024

Best-val checkpoints (best_clf / best_reg) of the data-scaling runs, evaluated
with expts/data-scaling/eval.py via eval-b200.sh (1xB200, il-lo): test split,
ctx 4096, items_per_task 1024, shuffle_seed 0 (same 1024-row subsample across
models). Scored by relbench's evaluator; `~` in the logs marks subsampled
tasks. Jobs 99567 (rt-j), 99576 (10pct), 99577 (32pct); checkpoints from
/dfs/user/ranjanr/share/relational-transformer/expts/data-scaling/.

## Classification (ROC-AUC, higher is better)

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-churn | 0.7789 | 0.8159 | 0.7964 |
| rel-amazon/user-churn | 0.6530 | 0.6910 | 0.6868 |
| rel-avito/user-clicks | 0.4518 | 0.5839 | 0.6716 |
| rel-avito/user-visits | 0.5514 | 0.5868 | 0.6011 |
| rel-event/user-ignore | 0.8111 | 0.8606 | 0.8536 |
| rel-event/user-repeat | 0.6957 | 0.6493 | 0.7563 |
| rel-f1/driver-dnf | 0.7777 | 0.7988 | 0.8082 |
| rel-f1/driver-top3 | 0.8492 | 0.8968 | 0.8818 |
| rel-hm/user-churn | 0.6144 | 0.6390 | 0.6368 |
| rel-stack/user-badge | 0.7216 | 0.6494 | 0.6990 |
| rel-stack/user-engagement | 0.9129 | 0.9110 | 0.9174 |
| rel-trial/study-outcome | 0.5872 | 0.6105 | 0.6297 |
| **mean** | **0.7004** | **0.7244** | **0.7449** |

## Regression (normalized MAE, lower is better)

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-ltv | 0.0775 | 0.0701 | 0.0749 |
| rel-amazon/user-ltv | 0.2985 | 0.2691 | 0.2832 |
| rel-avito/ad-ctr | 0.4439 | 0.4591 | 0.4143 |
| rel-event/user-attendance | 0.3909 | 0.4185 | 0.4174 |
| rel-f1/driver-position | 0.4925 | 0.4333 | 0.4046 |
| rel-hm/item-sales | 0.1276 | 0.0957 | 0.0933 |
| rel-stack/post-votes | 0.1757 | 0.1526 | 0.1480 |
| rel-trial/site-success | 0.4021 | 0.3411 | 0.3228 |
| rel-trial/study-adverse | 0.1581 | 0.1650 | 0.1557 |
| **mean** | **0.2852** | **0.2672** | **0.2571** |

Caveat: the 10pct column is from a run that failed at step 44000/100001 (best
ckpts reconstructed from val_metrics.jsonl, peaking at steps 6000/8000); the
other two trained the full 100001 steps.
