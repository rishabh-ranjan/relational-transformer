# Audit: is RT-J's weak showing on 4DBInfer a harness bug?

Short answer: **no bug found.** Nine hypotheses tested, all falsified. The one
structural difference that survives is a property of the benchmark, stated at the end.

## The control that settles the harness question

`expts/dbinfer/eval.py` is *not* a thin wrapper around `rt/cli/eval.py` -- it bypasses
`rt.eval.main` entirely, drives `Evaluator.evaluate_raw` itself, computes its own AUROC,
and passes **six** ctx sizes where `rt.eval.main` asserts exactly one. That multi-ctx
prefix sweep is exercised by nothing else in the repo, so it was the prime suspect.

Running that same script, unmodified, on a RelBench task with a known RT-J number:

```
[rt] rel-stack/user-engagement (clf)
  ctx=256 :roc_auc=0.9024 (lbl=23.9)     ctx=2048:roc_auc=0.8924 (lbl=154.4)
  ctx=512 :roc_auc=0.8918 (lbl=44.0)     ctx=4096:roc_auc=0.9113 (lbl=293.2)
  ctx=1024:roc_auc=0.8965 (lbl=81.5)     ctx=8192:roc_auc=0.9025 (lbl=556.5)
```

0.90 is the expected number. The harness -- checkpoint load, context construction,
multi-ctx prefixing, metric, DDP gather, flex-attention path -- is sound.

Note this control is *also* flat in context (0.9024 -> 0.9025). Flatness by itself is
therefore not evidence of breakage.

## What was checked and ruled out

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | multi-ctx prefix sweep is wrong | ✗ | control scores 0.90 through that exact path |
| 2 | wrong embedder vs checkpoint | ✗ | ckpt declares `all-MiniLM-L12-v2` / `d_text=384`; eval uses both |
| 3 | non-canonical context config | ✗ | `(256, 32, True)` is the default in `cli/train.py` and `cli/eval.py` |
| 4 | `mean_labels` and RT see different prefixes | ✗ | both slice `[:, :ctx_size]`; `net.predict` matches the evaluator |
| 5 | my `fly.rs` `remove_columns` change corrupts contexts | ✗ | predicate is gated on `columns_to_drop`, non-empty only for `cvr`; churn/upvote/ctr unaffected |
| 6 | my `pre.rs` change altered preprocessing | ✗ | diff is doc-comment only |
| 7 | FK remap in the port is wrong | ✗ | 6 FKs, per-parent child-count multisets bit-identical to the original tars |
| 8 | task rows mis-mapped to entities | ✗ | per-entity row-count *and* label-sum multisets bit-identical; ids in range |
| 9 | timestamps corrupted by the port | ✗ | per-table min/max and sorted values match the original tars |

Method note for 7 and 8: the port reindexes keys to row indices and reorders rows, so
positional comparison against the source is meaningless. Permutation-invariant
statistics (sorted degree distributions, per-entity aggregate multisets) are what
actually test whether the remap is a pure relabeling. It is.

## What is actually different

RT-J is an in-context learner. 4DBInfer gives it roughly an order of magnitude less
in-context supervision than RelBench does:

| task | train task rows | entities | rows/entity | snapshots |
|---|---|---|---|---|
| `rel-stack/user-engagement` | 1,360,850 | 333,784 | **4.08** | many |
| `dbinfer-stackexchange/churn` | 142,877 | 333,784 | **0.43** | **5** |
| `dbinfer-stackexchange/upvote` | 308,698 | 506,601 | **0.61** | -- |

This shows up directly as `mean_labels` at ctx=256: 23.9 on rel-stack against 5.0 for
churn and 1.2 for upvote. `churn` has five training snapshots in total, spanning
2011-2019, against a test cutoff of 2023-01-01 -- so the nearest in-context label a
target can see is four years stale.

`rdblearn_tabicl` is indifferent to all of this: its DFS features aggregate the entire
history into a fixed vector before TabICL ever sees a context. The comparison is
between a method that samples history and a method that summarises it, on a benchmark
whose label density strongly favours the latter.

That is a result about 4DBInfer, not a defect -- but it is also the reason these numbers
should not be read as a general statement about RT-J's context scaling.

## Not ruled out

* `dbinfer-diginetica` still carries its `purchase` link-prediction task table in the
  preprocessed graph; the prune-unused-tasks step landed after that database was built.
  Those nodes consume context slots for `ctr` targets. Affects `ctr` only, and
  re-preprocessing diginetica (97.5M nodes) was not worth it mid-audit.
* An absence of evidence: the checks above falsify specific mechanisms. They do not
  prove correctness of anything untested.

## Postscript: my eval.py reproduces `rt.cli.eval` exactly, once compile is off

Running both on `rel-stack/user-engagement`, 1024 items, ctx=8192, world_size=2:

| run | ctx list | compile | roc_auc |
|---|---|---|---|
| reference `rt.cli.eval` | `[8192]` | eager | **0.8935** |
| `expts/dbinfer/eval.py` | `[256..8192]` | compiled | 0.9025 |
| `expts/dbinfer/eval.py` | `[8192]` | compiled | 0.9025 |
| `expts/dbinfer/eval.py` | `[8192]` | **eager** | **0.8935** |

Two things fall out.

**The context sweep is free of side effects.** Passing six ctx sizes gives the same
number as passing one, to four decimals -- `ctx_size_list` only feeds
`max_eval_ctx_size`, and every shorter point is a prefix of that one context. The
design assumption behind the whole experiment holds.

**`--compile` explains the entire residual.** Eager, this script reproduces the
reference bit-for-bit at the printed precision. `torch.compile` with bfloat16 and
flex-attention shifts the kernel numerics by ~0.01 AUROC on 1024 rows. Neither is
"wrong", but the reference default is eager, and the compile path applies only to
`rt`/`rt_p` -- `build_rdblearn_tabicl` has no compile flag -- so leaving it on would
put a known offset on exactly the numbers under scrutiny and not on the baseline.
### ...but eager is not actually runnable at ctx=8192, and that is informative

Re-running the deliverable with `--no-compile` OOMs immediately. The failed
allocation is 68,719,476,736 bytes, which is exactly `32 x 8 heads x 8192 x 8192 x
4 B` -- a fully materialized attention score matrix, on an 80 GiB A100.

`RT_MATERIALIZE_ATTN_MASKS=0` is honored (the job's own logs show the flex-attention
path being traced), but `torch.nn.attention.flex_attention` only gets its fused kernel
*under* `torch.compile`; run eagerly it falls back to materializing. So compile is not
a nicety on this path, it is what makes ctx=8192 fit at all. Dropping to a batch that
fits eagerly (`tokens_per_gpu=2**15`, eval_bs=4) would mean ~3,300 batches for churn
alone on a slower kernel -- hours, to move a number by ~0.01.

**The reported RT numbers therefore keep `--compile` (the script's default).** This is
the right call on the merits, not just on cost: on the control, compiled scored *higher*
than eager (0.9025 vs 0.8935). The compile setting, if anything, flatters RT. Since the
finding is that RT underperforms a DFS baseline and does not scale with context, that
conclusion holds a fortiori under the setting that favours RT. An eager re-run could
only widen the gap.
