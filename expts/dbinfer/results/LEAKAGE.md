# Does `rdblearn_tabicl` leak?

Motivation: RT-J losing to DFS+TabICL by this margin is surprising enough that the
baseline deserves to be treated as the suspect, not the reference.

**Conclusion: no leak that could explain the gap.** Three independent tests below.

## 1. No single feature is near-perfectly predictive

A leaked label shows up as one feature with AUROC ~1.0. Per-feature AUROC against the
true test labels, taking `max(a, 1-a)` so an inverted feature cannot hide:

| task | features | n | **max single-feature AUROC** | >0.95 | >0.99 |
|---|---|---|---|---|---|
| `diginetica/ctr` | 25 | 6,616 | 0.6620 | 0 | 0 |
| `retailrocket/cvr` | 38 | 9,997 | 0.7432 | 0 | 0 |
| `stackexchange/churn` | 44 | 105,612 | 0.8130 | 0 | 0 |
| `stackexchange/upvote` | 34 | 38,588 | 0.8245 | 0 | 0 |

Not one feature above 0.95 anywhere.

## 2. The model scores about what its features are worth

If the label were leaking, the model would land near the leaking feature's AUROC
(~0.99). Instead it lands slightly above its *best* feature -- the signature of an
ordinary tabular model combining many weak-to-moderate features:

| task | best single feature | model (fulltest, ctx=8192) | lift |
|---|---|---|---|
| `ctr` | 0.6620 | 0.6836 | +0.02 |
| `cvr` | 0.7432 | 0.7880 | +0.04 |
| `churn` | 0.8130 | 0.8446 | +0.03 |
| `upvote` | 0.8245 | 0.8525 | +0.03 |

A +0.02..0.04 lift over the best of 25-44 features is what a good tabular model does.
It is not what leakage looks like.

## 3. Features respect each row's cutoff

DFS computed once, ignoring the per-row cutoff, would be a temporal leak that raises
many features mildly rather than any one to 1.0 -- so test 1 would not catch it.
`stackexchange/churn` has every one of its 88,164 val entities also present in test,
under two different cutoffs (val 2021-01-01, test 2023-01-01):

```
entities in both val and test              : 88,164
identical feature vectors across cutoffs   : 0.0000
mean fraction of feature entries differing : 0.3632
```

Zero rows carry the same vector across the two cutoffs. The featurizer is
cutoff-aware. (`include_cutoff_time=False` -- strict `<` -- is pinned in
`rel2tab/featurizers/rdblearn_featurizer.py` so an upstream default flip cannot
silently change it.)

## One real but immaterial defect

12 of 105,612 `churn` test rows belong to users whose `Users.CreationDate` is *after*
the test cutoff. Those users have no pre-cutoff history at all, yet 43% of their
feature entries are non-zero (against 75% for other rows), and none is entirely zero
-- consistent with DFS including the seed entity's own attributes, which for these
users are themselves post-cutoff facts.

It is a leak in principle. It is 0.011% of rows, it cannot move AUROC at the third
decimal, and it does not touch the other three tasks' conclusions.

## What this means

The baseline's advantage is real. It is not an artifact to be explained away, and
the RT-vs-baseline gap has to be explained on the modelling side -- see `AUDIT.md`,
which locates it in 4DBInfer supplying ~10x less in-context supervision per entity
than RelBench, which DFS aggregates are indifferent to.
