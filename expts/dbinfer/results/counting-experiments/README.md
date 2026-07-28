# Is the residual gap about counting? No.

`WHY_RT_TRAILS.md` argued the gap on the stackexchange tasks was DFS materialising
cardinality that RT cannot compute. Two experiments were run to test it. **The
counting hypothesis is refuted, and so is the follow-up hypothesis.**

## Experiment 2: drop every COUNT(...) feature from the baseline

Feature names now travel into `<table>_meta.json`, so the ablation selects columns by
what they are rather than by guessing. `churn` 44 -> 38 features, `upvote` 34 -> 27.

| task | full | COUNTs dropped | delta |
|---|---|---|---|
| `churn` | 0.8549 | 0.8455 | **-0.0094** |
| `upvote` | 0.8382 | 0.8350 | **-0.0032** |

Removing *every* count costs 0.003-0.009 against a 0.06-0.16 gap to RT. The baseline
does not need them. `mean_labels` is identical between the two arms, and the
full-feature arm reproduces the s3-threshold reference (`upvote` exactly, `churn`
0.8549 vs 0.8563 -- a ~0.001 wobble from TabICL's batch composition changing with the
task list, which is this baseline's run-to-run noise floor).

## Why: the strongest features are recency, and everything is redundant

| task | feature | AUROC |
|---|---|---|
| `churn` | `MAX(PostHistory.CreationDate)_epochtime_diff` | **0.8130** |
| | `MAX(Posts[OwnerUserId].CreationDate)_epochtime_diff` | 0.8103 |
| | `COUNT(PostHistory)` | 0.7506 |
| `upvote` | `COUNT(Vote)` | 0.8245 |
| | `MAX(Vote.CreationDate)_epochtime_diff` | 0.8180 |

`epochtime_diff` is `cutoff - most recent child timestamp` -- time since last activity,
the classic churn predictor, and on `churn` it beats every count. Count, max and mode
over the same child set all encode roughly the same underlying signal, which is exactly
why deleting one whole class of them barely moves the metric.

## The follow-up hypothesis is also wrong

Global datetime normalisation looked like a candidate: `pre.rs` uses a *single*
mean/std for every datetime cell in the database, so recency deltas might be
unresolvable. Measured, they are not:

```
datetime cells 5,295,849   span 5,326 days (14.6 yr)
single global normalizer std = 1,185 days
churn recency p25..p75 = 824..2576 days  =  1.48 SD
```

Comfortably resolvable. (An earlier version of this number was wrong by 1000x because
the port stores `timestamp[us]`, not `[ns]`.)

## What is actually left

Not a missing quantity, and not a resolution failure. DFS materialises aggregate
statistics over the *complete* child set before TabICL sees anything; RT has to derive
them in-context -- take a max over a set of rows, then difference it against the
target's own timestamp. The ingredients are present and legible in RT's context; the
reduction is the part it has to do itself, and nothing in the in-context setup trains
that.

This predicts experiment 1 (injecting cutoff-respecting counts as task-table columns)
will *not* close the gap, since it supplies one aggregate the baseline barely needs.
