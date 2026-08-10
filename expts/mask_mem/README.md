# What the dense attention masks cost

```
pixi run python expts/mask_mem/submit.py     # one a100, il-interactive, ~2 min
```

`rt.model.net` with `materialize_attn_masks=True` builds three `(B, S, S)` bool
masks and converts each to a `BlockMask`. At the pretraining shape -- 16 rows of
8192 tokens, the batch a `ctx_size=8192` step produces -- one such mask is 1 GiB,
and the pretraining run has now OOM'd twice in a training forward.

[`probe.py`](probe.py) times and measures peak allocation for three ways of
getting the same three block masks:

- `all_at_once` -- what `net.py` does: build all three dense masks, then convert
  all three, so every mask is alive while the last is built.
- `one_at_a_time` -- build one, convert it, drop it, then the next. A `BlockMask`
  indexes at 128-token block granularity, so it is orders of magnitude smaller
  than the dense mask it came from; only one dense mask need ever be live.
- `mask_mod` -- `materialize_attn_masks=False`, where `create_block_mask` samples
  the predicate at block granularity and no dense mask exists at all. The floor.

Results go to the job's log under
`/dfs/user/ranjanr/slurm-logs/.../expts/mask-mem`.

Delete this directory once the question is settled and the answer is in
`net.py`.
