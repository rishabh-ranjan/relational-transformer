# Why RT still trails DFS+TabICL on 4DBInfer

After the identifier fix (`results/identifier-policy/`), the gap is no longer uniform.
At the best arm (`s3-threshold`, ctx=8192, 1024-subsampled test):

| task | rt_p | rdblearn_tabicl | gap |
|---|---|---|---|
| `retailrocket/cvr` | 0.8094 | 0.8143 | **0.005 -- tied** |
| `stackexchange/churn` | 0.7963 | 0.8563 | 0.060 |
| `stackexchange/upvote` | 0.6757 | 0.8382 | **0.163** |
| `diginetica/ctr` | 0.4272 | 0.7217 | 0.295 (inverting; separate issue) |

So the question is not "why is RT worse" but "why is RT worse *on the stackexchange
tasks*".

## The discriminative signal on those tasks is a row count

`stackexchange/upvote`. The single best DFS feature scores 0.8245. It is a vote count:
computing `COUNT(Vote rows for this post with CreationDate < cutoff)` by hand scores
**0.8245**, matching to four decimals, and correlates 0.73 with that feature.

`stackexchange/churn`. Same shape. Pre-cutoff counts alone, per user:

| count | AUROC |
|---|---|
| `PostHistory` rows | 0.7506 |
| `Posts` rows | 0.7469 |
| `Comments` rows | 0.7414 |
| `Badges` rows | 0.7134 |

Against a best DFS feature of 0.8130 and a model score of 0.856 -- i.e. the model is
counts plus a little.

`retailrocket/cvr` is the task where RT ties, and it is not count-shaped: it turns on
which item and visitor are involved and their recent views, which are individual rows
RT can read directly.

## It is not an access problem -- RT has the rows

The obvious explanation would be that these neighbourhoods are too large to sample:
`bfs_width=32`, so a post with 200 votes would be truncated and the count destroyed.
**That is not what happens.** Pre-cutoff counts are tiny:

```
upvote, votes per post before cutoff:  median 0, p90 2, p95 4, p99 7
                                       fraction with >32:  0.0000
                                       fraction with  0:   0.5073
churn, per-user pre-cutoff counts:     fraction with >32:  0.011 - 0.040
```

Not one post in 38,588 has more than 32 pre-cutoff votes. The rows that carry the
signal fit in the context with room to spare, and half the discriminative work is just
"does this post have *any* votes yet".

So RT is given the rows and does not turn them into the count.

## What that means

The asymmetry is representational, not informational. DFS materialises cardinality as
a scalar feature before TabICL sees anything; RT would have to compute cardinality by
attending over a bag of rows, which is a weak operation for attention and one nothing
in its in-context setup teaches it to perform. On tasks whose label is a smooth
function of *which* rows exist, RT is competitive (`cvr`, tied). On tasks whose label
is a function of *how many* rows exist, it is not.

This also explains why the identifier fix helped where it did. Making a silent table
visible adds rows RT can attend to -- worth +0.12 on `cvr`, where identity matters --
but it does not give RT a counting operator, which is why `upvote` did not move
(0.697 -> 0.649) even though `PostTag` and `Vote` became visible.

## What would test this directly

1. **Give RT a count.** Add degree/count columns at preprocessing time (`n_votes`,
   `n_comments`, ... as of the row's timestamp) and re-run. If `upvote` jumps toward
   0.82, the diagnosis is confirmed and the fix is mechanical.
2. **Ablate the baseline's counts.** Drop `COUNT(...)` features from the DFS matrix
   and re-run `rdblearn_tabicl`. If it falls to RT's level, the two are then measured
   on equal footing.

(2) is the cleaner scientific test; (1) is the one that would improve RT.
