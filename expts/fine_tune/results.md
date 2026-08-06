# Results

## Classification (test ROC-AUC %, higher is better)

| dataset/task | dummy | dummy-per-entity | graphsage | lightgbm | rdblearn | relgnn | relgnn-es | relgt | tabpfn-rel | tabpfn-rel-no-text |
|---|---|---|---|---|---|---|---|---|---|---|
| rel-amazon/item-churn | 50.0 | 72.9 | **83.1** | 66.2 | 82.2 | 77.4 | 78.6 | 82.4 | 82.8 | 82.8 |
| rel-amazon/user-churn | 50.0 | 63.4 | **70.5** | 51.7 | 69.2 | 69.7 | 69.4 | 70.2 | 70.3 | 70.2 |
| rel-avito/user-clicks | 50.0 | 50.4 | 60.9 | 56.4 | **68.0** | 66.2 | 66.8 | 64.4 | 65.2 | 61.5 |
| rel-avito/user-visits | 50.0 | 60.3 | 66.6 | 52.9 | 66.4 | 65.2 | 64.9 | 66.2 | 66.8 | **66.9** |
| rel-event/user-ignore | 50.0 | 84.0 | 75.9 | 74.6 | 74.1 | 79.7 | 80.5 | 78.2 | **87.5** | 70.1 |
| rel-event/user-repeat | 50.0 | 75.2 | 78.5 | 75.4 | 77.6 | **78.5** | 75.5 | 73.4 | 76.4 | 76.9 |
| rel-f1/driver-dnf | 50.0 | 69.9 | 71.7 | **73.0** | 70.9 | 71.5 | 72.6 | 71.2 | 72.4 | 71.4 |
| rel-f1/driver-top3 | 50.0 | 55.7 | 72.6 | 73.9 | 78.6 | 77.8 | 75.9 | **81.1** | 77.7 | 79.3 |
| rel-hm/user-churn | 50.0 | 64.8 | 69.9 | 59.0 | 69.8 | 69.2 | 68.2 | 69.0 | 70.5 | **70.6** |
| rel-stack/user-badge | 50.0 | 78.9 | **88.9** | 66.0 | 82.6 | 61.1 | 62.1 | 57.4 | 88.1 | 86.3 |
| rel-stack/user-engagement | 50.0 | 82.7 | 90.6 | 81.2 | 89.7 | 90.6 | 90.5 | **90.7** | 90.6 | 90.6 |
| rel-trial/study-outcome | 50.0 | 50.0 | 68.6 | 71.5 | 73.2 | 69.2 | 65.7 | 66.8 | **75.5** | 73.1 |

## Regression (test nMAE %, lower is better)

| dataset/task | dummy | dummy-per-entity | graphsage | lightgbm | rdblearn | relgnn | relgnn-es | relgt | tabpfn-rel | tabpfn-rel-no-text |
|---|---|---|---|---|---|---|---|---|---|---|
| rel-amazon/item-ltv | 10.9 | 11.1 | 8.3 | 9.4 | 8.3 | 8.9 | 8.9 | 8.2 | **7.9** | 8.1 |
| rel-amazon/user-ltv | 29.2 | 30.3 | 25.1 | 29.2 | 25.0 | 26.1 | 25.3 | **25.0** | 25.1 | 25.0 |
| rel-avito/ad-ctr | 45.0 | 43.1 | 40.7 | 43.1 | 33.6 | 44.3 | 44.6 | 38.1 | **32.4** | 32.8 |
| rel-event/user-attendance | 34.4 | 35.2 | 32.0 | 34.3 | 31.0 | 32.2 | 31.9 | 34.2 | **30.6** | 31.3 |
| rel-f1/driver-position | 62.6 | 58.4 | 57.1 | 58.4 | 53.9 | 60.0 | 60.7 | 67.8 | 53.6 | **53.5** |
| rel-hm/item-sales | 15.4 | 15.8 | 11.1 | 15.2 | 12.9 | 11.5 | 11.4 | **10.7** | 12.2 | 12.4 |
| rel-stack/post-votes | 13.3 | 13.6 | **12.7** | 13.2 | 13.2 | 13.3 | 13.3 | 13.3 | 13.2 | 13.3 |
| rel-trial/site-success | 97.1 | 92.7 | **68.3** | 91.9 | 92.5 | 97.1 | 71.6 | 77.8 | 87.5 | 81.1 |
| rel-trial/study-adverse | 17.0 | 17.0 | 13.1 | 13.1 | 13.0 | 14.2 | 13.6 | 13.0 | **11.9** | 12.6 |

nMAE = MAE / std(train target), std from `stanford-star/relbench` (`regression_stds.json`).
