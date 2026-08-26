# Leaderboard ensemble (RT-J paper rerun)

RT-J's RelBench leaderboard submission and the paper's tuned+ensembled
per-task table: for every task, the **top-4 context configurations** by
validation score (from [`../tune`](../tune)) each run with
**4 context seeds** on the **full official test split**; the 16 raw
predictions are averaged per row, then denormalized (reg) / sigmoided (clf)
into prediction CSVs scored by RelBench's own evaluator.

## Running it

```bash
pixi run python -m expts.repaper.submit.submit   # 84 one-GPU units (21 tasks x 4 cfgs)
pixi run python -m expts.repaper.submit.reduce   # -> CSVs + results.json; prints next step

# validate + package (writes the leaderboard zips next to the CSVs)
pixi run python -m relbench.submit \
    ~/scratch/hf/relational-transformer/repaper/leaderboard/preds
```

The zips + validation report are the submission package; it stays under
`~/scratch/hf/relational-transformer/repaper/leaderboard/`. Submit
by opening the issue form on `rishabh-ranjan/relbench` and attaching the
zips, per that repo's README.

A unit is `expts.repaper.enscurve.run:main` (the ensemble runner) at
`n_seeds=4` on the full split; it resumes per seed, so preemption costs one
full-test pass at most. `reduce.py` refuses a task whose four units have not
all finished their 4 seeds, and `_emit_and_score`'s alignment guard
(`|dy|`/`cls=`) proves the node-index join against relbench's ground truth on
every task.
