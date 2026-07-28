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

## Experiment 1: inject cutoff-respecting counts into RT's context

Counts added as columns on the task table (leak-free: each row counts only child rows
strictly before *its own* cutoff). Node offsets verified unchanged across all 15
tables before the DFS features were carried over.

| method | task | before | +counts | delta |
|---|---|---|---|---|
| `rt` | `upvote` | 0.6488 | **0.8293** | **+0.1804** |
| `rt_p` | `upvote` | 0.6757 | **0.8344** | **+0.1587** |
| `rt_p` | `churn` | 0.7963 | **0.8407** | +0.0445 |
| `rt` | `churn` | 0.7232 | 0.6850 | -0.0382 |
| `rdblearn_tabicl` | `churn` | 0.8563 | 0.8525 | -0.0039 |
| `rdblearn_tabicl` | `upvote` | 0.8382 | 0.8411 | +0.0030 |

Gap to the baseline, best RT variant:

| task | before | after |
|---|---|---|
| `churn` | +0.0601 | **+0.0117** |
| `upvote` | +0.1624 | **+0.0067** |

It closes, and closes *despite* the injected columns costing context -- `mean_labels`
falls 139.38 -> 133.55 on `churn` and 53.86 -> 52.86 on `upvote`. The baseline moves
within noise, which is the control working: it already had these aggregates.

## Reconciling the two

They look contradictory and are not. I predicted, from experiment 2, that experiment 1
would fail. That prediction was wrong, and the error was in the question being asked:

* **Exp 2** asks whether the *baseline* needs counts. It does not -- it carries several
  interchangeable aggregates (count, max, mode, recency) over the same child set, so
  deleting one class leaves the signal intact.
* **Exp 1** asks whether *RT* needs an aggregate. It does -- it had none at all, and
  supplying one closes a 0.16 gap to 0.007.

So the conclusion is neither "counting is the differentiator" (exp 2 refutes it) nor
"aggregates do not matter" (exp 1 refutes that): **RT's shortfall on these tasks is the
absence of any precomputed aggregate reduction over the child set. Any sufficient
aggregate closes it; the baseline happens to carry several redundant ones.**

Note this is the same shape as the `identifier` fix: in both cases RT was missing
something derivable in principle from its context but not materialised in it.

`rt` on `churn` is the one regression (-0.0382) while `rt_p` gained +0.0445 on
identical data -- four injected columns displaced ~6 labelled rows of context and RT-J
did not convert them. Unexplained, and it does not disturb the headline.
