## Table 1 -- classification

mean AUROC over 4 clf task(s) (higher is better)

| method | 256 | lbl | 512 | lbl | 1024 | lbl | 2048 | lbl | 4096 | lbl | 8192 | lbl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rdblearn_tabicl | 0.6120 | 21.79 | 0.6692 | 49.11 | 0.6701 | 98.83 | 0.7183 | 190.38 | 0.7641 | 367.42 | 0.7916 | 711.81 |
| rt_p | 0.6610 | 21.79 | 0.6783 | 49.11 | 0.6801 | 98.83 | 0.6567 | 190.38 | 0.6579 | 367.42 | 0.6670 | 711.81 |
| **rt-j** | 0.5979 | 21.79 | 0.6216 | 49.11 | 0.5648 | 98.83 | 0.5906 | 190.38 | 0.5755 | 367.42 | 0.5577 | 711.81 |

  rdblearn_tabicl: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote
  rt_p: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote
  rt: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote

## Table 2 -- regression

(no reg tasks)

per task

| method | task | type | n | 256 | 512 | 1024 | 2048 | 4096 | 8192 | lbl@256 |
|---|---|---|---|---|---|---|---|---|---|---|
| rdblearn_tabicl | dbinfer-diginetica/ctr | clf | 1024 | 0.4948 | 0.4771 | 0.4481 | 0.5309 | 0.7208 | 0.6923 | 69.45 |
| rdblearn_tabicl | dbinfer-retailrocket/cvr | clf | 1024 | 0.5820 | 0.7332 | 0.6893 | 0.7621 | 0.7118 | 0.7819 | 11.54 |
| rdblearn_tabicl | dbinfer-stackexchange/churn | clf | 1024 | 0.8266 | 0.8350 | 0.8337 | 0.8218 | 0.8039 | 0.8476 | 4.96 |
| rdblearn_tabicl | dbinfer-stackexchange/upvote | clf | 1024 | 0.5448 | 0.6317 | 0.7092 | 0.7586 | 0.8199 | 0.8446 | 1.20 |
| rt | dbinfer-diginetica/ctr | clf | 1024 | 0.3820 | 0.4054 | 0.2893 | 0.3779 | 0.2458 | 0.1612 | 69.45 |
| rt | dbinfer-retailrocket/cvr | clf | 1024 | 0.6556 | 0.7265 | 0.6669 | 0.6713 | 0.6850 | 0.6827 | 11.54 |
| rt | dbinfer-stackexchange/churn | clf | 1024 | 0.7807 | 0.7551 | 0.6939 | 0.6822 | 0.7006 | 0.6905 | 4.96 |
| rt | dbinfer-stackexchange/upvote | clf | 1024 | 0.5735 | 0.5994 | 0.6092 | 0.6309 | 0.6705 | 0.6966 | 1.20 |
| rt_p | dbinfer-diginetica/ctr | clf | 1024 | 0.6152 | 0.7519 | 0.7050 | 0.4835 | 0.4740 | 0.5131 | 69.45 |
| rt_p | dbinfer-retailrocket/cvr | clf | 1024 | 0.5397 | 0.5588 | 0.5860 | 0.6881 | 0.6741 | 0.6922 | 11.54 |
| rt_p | dbinfer-stackexchange/churn | clf | 1024 | 0.8617 | 0.7884 | 0.8052 | 0.8113 | 0.8375 | 0.8098 | 4.96 |
| rt_p | dbinfer-stackexchange/upvote | clf | 1024 | 0.6273 | 0.6142 | 0.6240 | 0.6439 | 0.6461 | 0.6530 | 1.20 |


fairness check: same rows scored, and mean_labels agree, across all methods at every ctx
