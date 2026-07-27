# Data-scaling eval: full test split

Same setup as results-subsampled1024.md but with the full test split
(--eval.items-per-task 1000000000): best_clf on forecast-clf.json, best_reg on
forecast-reg.json, ctx 4096, 1xB200 each. Jobs 99594 (10pct), 99595 (32pct),
99596 (rt-j); ~13.5h wall per model.

## Classification (ROC-AUC, higher is better)

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-churn | 0.7714 | 0.7986 | 0.7798 |
| rel-amazon/user-churn | 0.6482 | 0.6812 | 0.6749 |
| rel-avito/user-clicks | 0.4648 | 0.5384 | 0.5191 |
| rel-avito/user-visits | 0.5196 | 0.5479 | 0.5864 |
| rel-event/user-ignore | 0.7977 | 0.8422 | 0.8380 |
| rel-event/user-repeat | 0.6957 | 0.6493 | 0.7563 |
| rel-f1/driver-dnf | 0.7777 | 0.7988 | 0.8082 |
| rel-f1/driver-top3 | 0.8492 | 0.8968 | 0.8818 |
| rel-hm/user-churn | 0.5799 | 0.6228 | 0.6076 |
| rel-stack/user-badge | 0.7945 | 0.7670 | 0.7682 |
| rel-stack/user-engagement | 0.8582 | 0.8532 | 0.8703 |
| rel-trial/study-outcome | 0.5872 | 0.6105 | 0.6297 |
| **mean** | **0.6953** | **0.7172** | **0.7267** |

## Regression (normalized MAE, lower is better)

| task | 10pct | 32pct | rt-j |
|---|---|---|---|
| rel-amazon/item-ltv | 0.0970 | 0.0855 | 0.0919 |
| rel-amazon/user-ltv | 0.3198 | 0.2879 | 0.2979 |
| rel-avito/ad-ctr | 0.4631 | 0.4746 | 0.4338 |
| rel-event/user-attendance | 0.3671 | 0.3963 | 0.3946 |
| rel-f1/driver-position | 0.4925 | 0.4333 | 0.4046 |
| rel-hm/item-sales | 0.1624 | 0.1275 | 0.1271 |
| rel-stack/post-votes | 0.1786 | 0.1523 | 0.1476 |
| rel-trial/site-success | 0.4015 | 0.3355 | 0.3231 |
| rel-trial/study-adverse | 0.1665 | 0.1721 | 0.1628 |
| **mean** | **0.2943** | **0.2739** | **0.2648** |

Same 10pct caveat as the subsampled file: that run failed at step 44000/100001
with best-val ckpts from steps 6000/8000; 32pct and rt-j trained to 100001.
