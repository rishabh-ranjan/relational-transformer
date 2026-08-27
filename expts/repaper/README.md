# Regenerating every result of the RT-J paper

The experiments under this directory are the paper's results sections, end to
end: from a pretrained RT-J checkpoint pair to the wandb runs the paper's
figure scripts read, the two generated appendix tables, and the RelBench
leaderboard submission package. This file is the assignment and its order of
operations; each subdirectory's README owns that experiment's commands,
protocol and measured runtimes. [`../README.md`](../README.md) is how jobs
are submitted, allocated and watched here, and it applies to every job below.

| directory | produces |
|---|---|
| [`baselines/`](baselines/) | the baselines' feature blobs, FAISS indices, TabICL checkpoints, relbench cache |
| [`scaling/`](scaling/) | context-scaling curves: RT-J + 4 baselines (full test and subsampled), retriever and semantics ablations |
| [`tune/`](tune/) | per-task context grid on validation -> `tuned_configs.json` |
| [`enscurve/`](enscurve/) | ensemble-size curves, default and tuned context |
| [`valtest/`](valtest/) | default-vs-tuned appendix table |
| [`submit/`](submit/) | top-4-configs x 4-seeds full-test ensemble: leaderboard CSVs and table |
| [`pretrain_abl/`](pretrain_abl/) | five one-knob-off variants of the base pretraining run (masking rate, task mix) |

## 0. Set up the round

1. **Repos.** This checkout (`rishabh-ranjan/relational-transformer`, `main`)
   and the paper (`rishabh-ranjan/overleaf-rtj`, branch `rr-main`, cloned at
   `~/clones/rishabh-ranjan/overleaf-rtj`; `git pull` if it exists). Everything
   is committed before it is submitted and pushed often.
2. **Checkpoint.** `CKPT_CLF` and `CKPT_REG` are two directories, each holding
   the `config.json` + `model.safetensors` that
   `rt.model.RelationalTransformer.from_pretrained` loads, on a path every
   node can read (compute nodes have no Hub access). A Hub release is mirrored
   once with `huggingface_hub.snapshot_download("<org>/<repo>",
   local_dir="~/scratch/hf/<org>/<repo>")`; for `stanford-star/rt-j`
   the two directories are its `classification/` and `regression/`.
3. **Config.** Edit [`config.py`](config.py): `CKPT_CLF` / `CKPT_REG` to that
   pair, `RUN_TAG` to today's date. Set the same `RUN_TAG` in the paper repo's
   `gen/__init__.py`. Commit both. Nothing else names a checkpoint or a wandb
   project.
4. **Clear the previous round's checkpoint-dependent outputs.** Every job
   no-ops on an output that already exists, so stale results would otherwise
   count as finished:

   ```bash
   R=~/scratch/ckpts/rtv2
   rm -rf $R/repaper-scaling/fulltest/rt $R/repaper-scaling/subsampled/rt \
          $R/repaper-scaling/abl $R/repaper-enscurve $R/repaper-valtest \
          $R/repaper-submit $R/*-repaper-tune
   S=~/scratch/hf/relational-transformer/repaper
   rm -rf $S/features/*/rt_features $S/vector_db/rt $S/leaderboard
   ```

   What stays valid across checkpoints (same preprocessed data, same default
   context): the SQL and RDBLearn feature blobs and their FAISS indices, the
   classic-relbench cache, the TabICL checkpoints, the semantics-ablated data
   copy, and the eight baseline arms under
   `repaper-scaling/{fulltest,subsampled}/{sql,rdblearn}_{tabicl,lgbm}` --
   their contexts depend only on the sampler. If the preprocessed data or the
   default context changed too: `rm -rf $R/repaper-scaling $S/features
   $S/vector_db $S/relbench-preprocessed-nosem`.
5. **One-time fetches** (login node; compute nodes have no internet), if the
   shared directory is empty: [`baselines/README.md`](baselines/README.md)
   steps 0-2 -- TabICL checkpoints, the classic-relbench cache,
   `pixi run install-rdblearn`, `check_alignment`.

## 1. Submit, in dependency order

Each stage is `pixi run python -m expts.repaper.<experiment>.submit` after
uncommenting the stage's loop in that script (and commenting out the others)
and writing the resource plan against the live cluster. Every job is idempotent per output
file, so a script is resubmitted to fill gaps.

| stage | experiment | waits for |
|---|---|---|
| features | `baselines` (sql, rdblearn, rt) | fetches |
| FAISS indices | `baselines` (the vector-db loop) | a feature set's 21 blobs |
| semantics-ablated data | `scaling` (the `make_nosem_data` call) | nothing |
| RT arms + ablations | `scaling` (rt, rand, bfs*) | checkpoint only |
| `nosem` arm | `scaling` | the ablated data |
| baseline arms | `scaling` (sql/rdblearn x tabicl/lgbm) | feature blobs |
| vdb arms | `scaling` (vdb_rdblearn, vdb_rt) | FAISS indices |
| tuning grid | `tune` | checkpoint only -- the critical path |
| ensemble curves, default | `enscurve` | checkpoint only |
| `tuned_configs.json` | `tune` (`collect`, commit) | all 21 grids |
| ensemble curves, tuned | `enscurve` | `tuned_configs.json` |
| default-vs-tuned table | `valtest` (`collect` only) | both `enscurve` variants |
| leaderboard ensemble | `submit` | `tuned_configs.json` |
| pretraining ablations | `pretrain_abl` | nothing; lowest priority |

Submit the checkpoint-only stages together; they are the bulk of the GPU work
and the tuning grids and the rel-amazon full-test passes are their longest
poles, so that is where the high tiers go. In the 2026-08-19 round the
tuning grids and the leaderboard units were not submitted here: `expts/icl`
had just run both for RT-J on the same checkpoint pair, data and protocol,
and `tune/collect.py` and `submit/reduce.py` read its directories by path
(each README says which); `valtest` submits nothing in any round -- its table
is the single-seed point of the two ensemble curves. Each README's "Measured runtimes"
is what the last round measured on this cluster; project the round's ETA from
the first finished jobs of this one and fill those sections in. A pretraining
ablation is days on an exclusive node and stops itself.

## 2. Reduce and regenerate

When an arm is complete (21 task files), its reduce logs the wandb run the
paper reads: `scaling/reduce.py` (18 runs over the fulltest, subsampled and
ablation projects), `enscurve/reduce.py`, `valtest/collect.py`,
`submit/reduce.py`, then the leaderboard validation and packaging in
[`submit/README.md`](submit/README.md). Commit `tune/tuned_configs.json` and
`valtest/results.json`.

Then the paper: its `CLAUDE.md` section "Regenerating all results" is the
order of figure and table scripts, the compile, and the pass over the numbers
quoted in the prose. Inspect every PDF; push to `rr-main`.

The round is done when every figure and table reflects the new runs, the
paper compiles, the leaderboard package is written, and no job of the round
is left in the queue.

## 3. What to know before it bites

- A FAISS on-disk IVF index (every task table over 50k rows) bakes the
  absolute path of its `.ivfdata` file at build time, so the `vector_db`
  indices survive a checkpoint change but not a move of `SHARE`: after a
  move, `rm -rf $S/vector_db` and rebuild them (`baselines/submit.py`, the
  vecdb loop), or every `vdb_*` pass on a large table dies at index load.
- The semantics-ablated data is a directory of symlinks into `PRE_DIR` plus
  deranged embedding files; a moved `PRE_DIR` leaves the links dangling and
  every `abl/nosem` job crashing at start. `make_nosem_data` relinks, so
  `scaling/submit.py` submits it first and the `nosem` arm `after=` it.
- `db_cutoff=None` everywhere: per-row temporal masking is the only trim
  (`"test"` would resolve every Join source through the Hub and crash on
  databases without a test timestamp). The base pretraining run and its
  ablations pass `None` too.
- What a preemption costs: tuning resumes per grid entry, ensemble and
  leaderboard units per seed, pretraining from a checkpoint 20 minutes old;
  scaling and featurize jobs restart their task from the top, and the biggest
  full-test tasks are hours.
- The cpu-only partition (`il-cpu`) admits one QOS whose whole per-user
  budget a standing dev-node allocation holds; cpu work runs as zero-gres
  `il-lo` jobs on the `il` partition instead.
- A 1-gpu ampere job is capped at 252154M of memory by the submit plugin.
- Never pin a sweep to `hyperturing2`: its future slots are back-fill-claimed
  by higher-priority `il-lo` work a day out and pinned jobs park on
  `ReqNodeNotAvail`. `hyperturing1`'s cards throw uncorrectable ECC errors
  ([`../preprocess/README.md`](../preprocess/README.md)).
