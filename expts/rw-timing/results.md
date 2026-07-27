# rw-timing results

RT-J inference timed on 1x B200 (blackwell1, il-lo) via `timing.py`:
rel-f1/driver-dnf test split (702 items), ctx_size=8192, batch_size=1,
local_ctx_size=256, bfs_width=32, prefer_latest=true, 3 warmup + 10 timed
steps. Context construction (rustler, CPU) runs synchronously in the main
process; the forward pass (H2D copy + GPU compute) is bracketed by
`torch.cuda.synchronize()`, so the two steps never overlap.

## Per-step timing, ms/item (mean ± std)

| (num_walks, walk_length) | context construction | forward pass | total |
|---|---|---|---|
| (10000, 20) | 24.84 ± 4.60 | 26.00 ± 0.10 | 50.84 ± 4.63 |
| (1000, 10)  |  4.30 ± 0.39 | 26.47 ± 0.17 | 30.77 ± 0.30 |

Shrinking the walk config cuts context construction ~6x; the forward pass is
unchanged, so total drops from ~51 to ~31 ms/item.

Jobs: 99912 (10000, 20), 99921 (1000, 10).

## Full test-set eval

Job 99942 (`eval-b200.sh`): rt.cli.eval at default batch size
(tokens_per_gpu=2^18, bs=32 @ ctx 8192) under both walk configs; RT-J
classification checkpoint on the clf tasks, regression checkpoint on the reg
tasks. All rel-f1, rel-trial, and rel-event forecast tasks, test split.

### Classification, roc_auc (higher is better)

| task | (10000, 20) | (1000, 10) |
|---|---|---|
| rel-f1/driver-dnf | 0.8090 | 0.8118 |
| rel-f1/driver-top3 | 0.9108 | 0.8995 |
| rel-event/user-ignore | 0.8325 | 0.8362 |
| rel-event/user-repeat | 0.7521 | 0.7417 |
| rel-trial/study-outcome | 0.6369 | 0.5201 |

### Regression, nmae (lower is better)

| task | (10000, 20) | (1000, 10) |
|---|---|---|
| rel-f1/driver-position | 0.4137 | 0.4281 |
| rel-event/user-attendance | 0.3578 | 0.3601 |
| rel-trial/site-success | 0.3141 | 0.3856 |
| rel-trial/study-adverse | 0.1543 | 0.1681 |
| mean | 0.3100 | 0.3355 |

### Wall time per eval run (whole task set, incl. data loading)

| run | (10000, 20) | (1000, 10) |
|---|---|---|
| clf (5 tasks) | 133 s | 129 s |
| reg (4 tasks) | 732 s | 684 s |

The rel-f1 and rel-event tasks are barely affected by the smaller walk
config; the rel-trial tasks degrade noticeably (study-outcome roc_auc
0.637 -> 0.520, site-success nmae 0.314 -> 0.386).
