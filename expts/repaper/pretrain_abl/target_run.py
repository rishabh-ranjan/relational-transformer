import wandb

api = wandb.Api()
targets: dict[str, float] = {}
for run in api.runs("rtv2/2026-09-09_pretrain"):
    if run.name.startswith("rt-j-classification"):
        pick = "auroc"
    elif run.name.startswith("rt-j-regression"):
        pick = "nmae"
    else:
        continue
    for k, v in run.summary.items():
        if k.startswith(f"{pick}/val/") and isinstance(v, (int, float)):
            targets[f"target/{k}"] = float(v)

assert targets
run = wandb.init(
    entity="rtv2",
    project="2026-09-09_pretrain",
    name="target",
    id="rt-j-target",
    resume="allow",
)
for step in (0, 2**15):
    run.log({"step": step, **targets})
run.finish()
print(f"logged {len(targets)} target series")
