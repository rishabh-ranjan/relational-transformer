# Reading these numbers

## Only 2 of the 4 tasks are a valid RT-J measurement

| task | valid for rt-j? | why |
|---|---|---|
| `stackexchange/churn` | yes | not in rt-j's training set, no label structure to exploit |
| `stackexchange/upvote` | yes | same |
| `retailrocket/cvr` | **no -- contaminated** | `join-retailrocket` is in `rt-j.json` (40 task-pairs), and it is the same database: bit-equal timestamp sets, `item_key == itemid` at 100% |
| `diginetica/ctr` | **no -- leaky** | label is exactly recoverable from same-timestamp sibling labels (below) |

The contamination is not a near-miss. RT-J was trained on `join-retailrocket`, which the
audit that opened this work established is `dbinfer-retailrocket` under another name. Its
`cvr` number measures partial memorisation, not transfer. `diginetica` and `stackexchange`
appear nowhere in `rt-j.json`.

Neither caveat touches `rdblearn_tabicl`: TabICL is trained on synthetic priors, and the
DFS features are recomputed here from scratch.

**So the headline RT-J comparison is the two `stackexchange` tasks.** The 4-task mean is
kept below only because it is what the campaign was specified to produce.

## `dbinfer-diginetica/ctr` does not measure modeling quality

RT-J scores 0.382 AUROC at ctx=256 and falls monotonically to **0.161** at ctx=8192.
An AUROC that far below 0.5, falling as context grows, is not a weak model -- it is a
model reading a signal backwards, more confidently the more of it it sees.

The task's own rows are the signal. Its test split:

| | |
|---|---|
| rows | 6,616 |
| distinct timestamps | 43 |
| rows per timestamp group | 153.9 mean, 200 max |
| positives per group | 1.5 mean |
| base positive rate | 0.98% |

Every row in a `queryId` group shares one timestamp -- they are the items displayed
for a single query -- and every group contains at least one click. So, on the test
split:

```
P(clicked = 1 | some sibling in my group is clicked) = 0.006
P(clicked = 1 | no sibling in my group is clicked)   = 1.000
```

The second line is not a correlation, it is an identity. A target whose visible
sibling labels are all 0 *is* the click, with certainty. The label is exactly
recoverable from the other labels in the context.

This inverts the usual in-context assumption. A model that treats nearby labels as
positively predictive -- which is what an ICL model does by default -- gets the sign
wrong on every group, and gets it wrong harder as more siblings enter the context.
That is precisely the 0.38 → 0.16 curve. A model that recognised the structure would
score ~1.0. Neither number says anything about relational modeling.

`rdblearn_tabicl` is unaffected only because it never sees task-row labels: its
features come from DFS aggregates over the database under a strict `<` cutoff.

The other three tasks do not have this structure:

| task | distinct ts | rows per ts group | positives per group |
|---|---|---|---|
| `diginetica/ctr` | 43 | 153.9 | 1.5 |
| `retailrocket/cvr` | 9,990 / 9,997 rows | 1.0 | 0.0 |
| `stackexchange/churn` | 1 | 105,612 | 3,929 |
| `stackexchange/upvote` | 38,506 / 38,588 rows | 1.0 | 0.7 |

`cvr` and `upvote` have one row per timestamp, so there are no siblings at all.
`churn` puts the whole split at one timestamp, so sibling labels carry the base rate
and nothing else -- an ordinary ICL prior, not a leak. `ctr` is the only task where a
group is small enough to be visible whole *and* carries a hard constraint on its
label sum.

## The mean with and without it

```
mean AUROC, 4 tasks             256      512     1024     2048     4096     8192
  rdblearn_tabicl            0.6120   0.6692   0.6701   0.7183   0.7641   0.7916
  rt-j                       0.5979   0.6216   0.5648   0.5906   0.5755   0.5577

mean AUROC, 3 tasks (no ctr)    256      512     1024     2048     4096     8192
  rdblearn_tabicl            0.6511   0.7333   0.7441   0.7808   0.7785   0.8247
  rt-j                       0.6699   0.6937   0.6567   0.6615   0.6854   0.6899
```

The headline conclusion survives the exclusion -- the baseline scales with context and
RT-J does not -- but its size does not. Including `ctr`, RT-J appears to *degrade*
with context (0.598 → 0.558); excluding it, RT-J is flat (0.670 → 0.690) and starts
*above* the baseline at ctx=256. Only the second is a statement about context scaling.

## Whether same-timestamp label visibility should be allowed at all

`Evaluator` hides only the target's own label (`node_idxs != target_node`); every
other task row in the context shows its label, including rows sharing the target's
timestamp. For `ctr` those rows are not available at prediction time in any real
deployment -- you do not know which other items in the same query were clicked.

This is the same class of leak as the `remove_columns` case, which was fixed in
`rustler/fly.rs` by dropping the column on rows sharing the target's timestamp. The
difference is that there the leaking column was a database column, and here it is the
task's own label column. The fix would be symmetric -- withhold task-row labels at the
target's timestamp, not just the target's own -- but it changes what every task's
context contains and what `mean_labels` counts, so it is a decision about the
benchmark, not a bug fix. Flagging rather than doing.
