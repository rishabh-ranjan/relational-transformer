# Pretraining ablations (RT-J paper rerun)

The two pretraining-ablation figures: multi-cell masking rate and pretraining
task mix. Five full-scale runs, each differing from the released pretraining
run (the base arm of both figures, wandb `rtv2/2026-08-07-pretrain` run
`rt-j`) in exactly the ablated knob, plus 10k-step early-stop patience so an
arm stops spending nodes once its val curve flattens. The base run predates
the `db_cutoff` knob, so the arms pass `None` -- the behavior it actually ran
with.

```bash
pixi run python -m expts.repaper_pretrain_abl.submit               # all 5 arms
pixi run python -m expts.repaper_pretrain_abl.submit <arm> <run_id>  # resume one
```

One 8xA100 node per arm, `--exclusive`, on the preemptible `il-lo` (the
non-preemptible `il` fits one node under the 10-a100 cap and the eval sweeps
are spending it); preemption and the wall clock both requeue and resume from
the run's checkpoint. **These are the lowest-priority jobs of the rerun:
submit them once the eval sweeps have drained**, or they compete with dozens
of one-GPU eval jobs for the same ampere pool.

The curves the figures read: `swa/nmae/val/mean` and `swa/auroc/val/mean`
against `step`, in `rtv2/2026-08-19-repaper-pretrain-abl` (runs `mask0`,
`mask25`, `mask75`, `mix-forecast`, `mix-autocomplete`) and the base run in
`rtv2/2026-08-07-pretrain`.
