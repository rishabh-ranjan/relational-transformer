# Identifier-policy ablation

ctx=8192, 1024-subsampled TEST, default (256,32,True) context config.

| strategy | method | ctr | cvr | churn | upvote | mean | mean (no ctr) |
|---|---|---|---|---|---|---|---|
| s0-none | rdblearn_tabicl | 0.6923 | 0.7819 | 0.8476 | 0.8446 | 0.7916 | 0.8247 |
| s0-none | rt_p | 0.5131 | 0.6922 | 0.8098 | 0.6530 | 0.6670 | 0.7183 |
| s0-none | rt | 0.1612 | 0.6827 | 0.6905 | 0.6966 | 0.5577 | 0.6899 |
| s1-empty | rdblearn_tabicl | 0.6795 | 0.8357 | 0.8477 | 0.8438 | 0.8017 | 0.8424 |
| s1-empty | rt_p | 0.3869 | 0.8240 | 0.8019 | 0.6495 | 0.6656 | 0.7585 |
| s1-empty | rt | 0.1795 | 0.8007 | 0.6954 | 0.6462 | 0.5804 | 0.7141 |
| s2-emptytime | rdblearn_tabicl | 0.5779 | 0.8386 | 0.8477 | 0.8438 | 0.7770 | 0.8434 |
| s2-emptytime | rt_p | 0.3494 | 0.8299 | 0.8019 | 0.6495 | 0.6576 | 0.7604 |
| s2-emptytime | rt | 0.3283 | 0.7614 | 0.6954 | 0.6462 | 0.6078 | 0.7010 |
| s3-threshold | rdblearn_tabicl | 0.7217 | 0.8143 | 0.8563 | 0.8382 | 0.8076 | 0.8363 |
| s3-threshold | rt_p | 0.4272 | 0.8094 | 0.7963 | 0.6757 | 0.6771 | 0.7605 |
| s3-threshold | rt | 0.2942 | 0.7805 | 0.7232 | 0.6488 | 0.6117 | 0.7175 |

## mean_labels @8192 (identical across the 3 methods within each strategy)

| strategy | ctr | cvr | churn | upvote |
|---|---|---|---|---|
| s0-none | 2267.14 | 380.74 | 142.93 | 56.42 |
| s1-empty | 2257.69 | 349.33 | 141.17 | 54.30 |
| s2-emptytime | 2155.78 | 347.91 | 141.17 | 54.30 |
| s3-threshold | 2069.71 | 341.47 | 139.38 | 53.86 |
