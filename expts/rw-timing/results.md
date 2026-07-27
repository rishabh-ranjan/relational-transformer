# rw-timing results

RT-J inference timed on 1x B200 (blackwell1, il-lo) via `timing.py`:
rel-f1/driver-dnf test split (702 items), ctx_size=8192, batch_size=1,
local_ctx_size=256, bfs_width=32, prefer_latest=true, 3 warmup + 10 timed
steps. Context construction (rustler, CPU) runs synchronously in the main
process; the forward pass (H2D copy + GPU compute) is bracketed by
`torch.cuda.synchronize()`, so the two steps never overlap.

## Per-step timing, ms/item (mean ± std over 10 steps)

| (num_walks, walk_length) | context construction | forward pass | total |
|---|---|---|---|
| (10000, 20) | 24.84 ± 4.60 | 26.00 ± 0.10 | 50.84 ± 4.63 |
| (10000, 10) | 10.80 ± 2.20 | 26.17 ± 0.10 | 36.97 ± 2.17 |
| (1000, 20)  |  6.05 ± 0.75 | 25.99 ± 0.12 | 32.04 ± 0.71 |
| (1000, 10)  |  4.30 ± 0.39 | 26.47 ± 0.17 | 30.77 ± 0.30 |

Context construction scales with both knobs (num_walks matters more); the
forward pass is constant (~26 ms/item).

Jobs: 99912, 100044, 100045, 99921.

## Full test-set eval

Job 100046 (`eval-b200.sh`): rt.cli.eval at default batch size
(tokens_per_gpu=2^18, bs=32 @ ctx 8192), 4 walk configs x 3 context seeds
(0, 1, 2); RT-J classification checkpoint on the clf tasks, regression
checkpoint on the reg tasks. All rel-f1, rel-trial, and rel-event forecast
tasks, test split. Cells are mean ± std over the 3 seeds.

| task | metric | (10000, 20) | (10000, 10) | (1000, 20) | (1000, 10) |
|---|---|---|---|---|---|
| rel-f1/driver-dnf | roc_auc ^ | 0.8137 ± 0.0041 | 0.8148 ± 0.0033 | 0.8129 ± 0.0041 | 0.8136 ± 0.0027 |
| rel-f1/driver-top3 | roc_auc ^ | 0.9103 ± 0.0007 | 0.9084 ± 0.0034 | 0.9071 ± 0.0034 | 0.9022 ± 0.0035 |
| rel-event/user-ignore | roc_auc ^ | 0.8331 ± 0.0030 | 0.8292 ± 0.0033 | 0.8280 ± 0.0007 | 0.8305 ± 0.0050 |
| rel-event/user-repeat | roc_auc ^ | 0.7433 ± 0.0174 | 0.7346 ± 0.0116 | 0.7380 ± 0.0080 | 0.7339 ± 0.0138 |
| rel-trial/study-outcome | roc_auc ^ | 0.6195 ± 0.0152 | 0.5664 ± 0.0079 | 0.5395 ± 0.0074 | 0.5162 ± 0.0101 |
| rel-f1/driver-position | nmae v | 0.4143 ± 0.0006 | 0.4208 ± 0.0007 | 0.4239 ± 0.0011 | 0.4280 ± 0.0012 |
| rel-event/user-attendance | nmae v | 0.3568 ± 0.0009 | 0.3578 ± 0.0013 | 0.3584 ± 0.0011 | 0.3587 ± 0.0017 |
| rel-trial/site-success | nmae v | 0.3142 ± 0.0009 | 0.3127 ± 0.0005 | 0.3562 ± 0.0001 | 0.3866 ± 0.0015 |
| rel-trial/study-adverse | nmae v | 0.1550 ± 0.0006 | 0.1594 ± 0.0004 | 0.1627 ± 0.0008 | 0.1683 ± 0.0002 |

rel-f1 and rel-event are essentially insensitive to cheaper walk configs
(differences within ~1-2 std). rel-trial degrades monotonically as walks
shrink: study-outcome roc_auc 0.62 -> 0.52 and site-success nmae
0.314 -> 0.387; study-adverse and driver-position degrade slightly but
consistently. num_walks=1000 is the main driver of the rel-trial drop.
