# Pretraining ablations (RT-J paper rerun)

The two pretraining-ablation figures: multi-cell masking rate and pretraining
task mix. Five runs, each differing from the round's base pretraining run in
exactly the ablated knob, plus 10k-step early-stop patience so an arm stops
spending nodes once its val curve flattens.

The base run is [`expts/pretrain/submit_marlowe.py`](../../pretrain/submit_marlowe.py):
its `args()` is imported here and each arm is `args() | ARMS[arm]`, so the two
cannot drift. The base logs to `rtv2/<RUN_TAG>-repaper-pretrain` as run
`base`; the arms to `rtv2/<RUN_TAG>-repaper-pretrain-abl` as `mask0`, `mask25`,
`mask75`, `mix-forecast`, `mix-autocomplete`. The task-mix lists beside this
file are the base's `all_5gb_cutoff.json` intersected with the Join's
`forecast` / `autocomplete` families (4098 / 9145 of 13243 pairs).

Write `ARM`, `RUN_ID` (`None` starts the arm, a run_id resumes it) and
`INSIDE` (a held allocation, or `None` to queue) into
[`submit.py`](submit.py), commit, then

```bash
pixi run python -m expts.repaper.pretrain_abl.submit
```

The base and the arms run one at a time on Marlowe in held 4-node
allocations (`expts/pretrain/README.md`), the arms after the base and each
other in whatever order the holders land.

The curves the figures read: `swa/nmae/val/mean` and `swa/auroc/val/mean`
against `step`; the paper's `gen/__init__.py` names the two projects and the
base run.
