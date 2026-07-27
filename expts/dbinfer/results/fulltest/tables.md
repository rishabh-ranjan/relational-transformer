## Table 1 -- classification

mean AUROC over 4 clf task(s) (higher is better)

| method | 256 | lbl | 512 | lbl | 1024 | lbl | 2048 | lbl | 4096 | lbl | 8192 | lbl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rdblearn_tabicl | 0.5995 | 21.72 | 0.6652 | 49.56 | 0.7014 | 99.19 | 0.7442 | 191.10 | 0.7781 | 368.11 | 0.7922 | 712.15 |
| rt_p | 0.5878 | 21.72 | 0.6012 | 49.56 | 0.6049 | 99.19 | 0.5998 | 191.10 | 0.6084 | 368.11 | 0.6366 | 712.15 |
| **rt-j** | 0.5751 | 21.72 | 0.5793 | 49.56 | 0.5727 | 99.19 | 0.5720 | 191.10 | 0.5820 | 368.11 | 0.5891 | 712.15 |

  rdblearn_tabicl: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote
  rt_p: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote
  rt: n=4  dbinfer-diginetica/ctr, dbinfer-retailrocket/cvr, dbinfer-stackexchange/churn, dbinfer-stackexchange/upvote

## Table 2 -- regression

(no reg tasks)

per task

| method | task | type | n | 256 | 512 | 1024 | 2048 | 4096 | 8192 | lbl@256 |
|---|---|---|---|---|---|---|---|---|---|---|
| rdblearn_tabicl | dbinfer-diginetica/ctr | clf | 6616 | 0.4627 | 0.5307 | 0.5755 | 0.6341 | 0.6880 | 0.6836 | 69.16 |
| rdblearn_tabicl | dbinfer-retailrocket/cvr | clf | 9997 | 0.6126 | 0.7502 | 0.7416 | 0.7720 | 0.7839 | 0.7880 | 11.59 |
| rdblearn_tabicl | dbinfer-stackexchange/churn | clf | 105612 | 0.7885 | 0.7814 | 0.8166 | 0.8254 | 0.8289 | 0.8446 | 4.91 |
| rdblearn_tabicl | dbinfer-stackexchange/upvote | clf | 38588 | 0.5340 | 0.5984 | 0.6721 | 0.7451 | 0.8118 | 0.8525 | 1.21 |
| rt | dbinfer-diginetica/ctr | clf | 6616 | 0.4372 | 0.4002 | 0.3449 | 0.3288 | 0.3215 | 0.2964 | 69.16 |
| rt | dbinfer-retailrocket/cvr | clf | 9997 | 0.6067 | 0.6559 | 0.6626 | 0.6330 | 0.6248 | 0.6233 | 11.59 |
| rt | dbinfer-stackexchange/churn | clf | 105612 | 0.7219 | 0.6908 | 0.6965 | 0.7045 | 0.7110 | 0.7233 | 4.91 |
| rt | dbinfer-stackexchange/upvote | clf | 38588 | 0.5345 | 0.5703 | 0.5867 | 0.6218 | 0.6706 | 0.7133 | 1.21 |
| rt_p | dbinfer-diginetica/ctr | clf | 6616 | 0.5250 | 0.5861 | 0.5893 | 0.5054 | 0.4829 | 0.5512 | 69.16 |
| rt_p | dbinfer-retailrocket/cvr | clf | 9997 | 0.4347 | 0.4690 | 0.4824 | 0.5285 | 0.5583 | 0.5697 | 11.59 |
| rt_p | dbinfer-stackexchange/churn | clf | 105612 | 0.8036 | 0.7703 | 0.7540 | 0.7471 | 0.7495 | 0.7566 | 4.91 |
| rt_p | dbinfer-stackexchange/upvote | clf | 38588 | 0.5881 | 0.5795 | 0.5939 | 0.6181 | 0.6430 | 0.6691 | 1.21 |


fairness check: same rows scored, and mean_labels agree, across all methods at every ctx
