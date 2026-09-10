import wandb

api = wandb.Api()
targets: dict[str, float] = {}
for run in api.runs("rtv2/2026-08-19-repaper-pretrain-abl"):
    if run.name.startswith("rt-plurel-classification"):
        pick = "auroc"
    elif run.name.startswith("rt-plurel-regression"):
        pick = "nmae"
    else:
        continue
    for k, v in run.summary.items():
        if k.startswith(f"{pick}/val/") and isinstance(v, (int, float)):
            targets[f"target/{k}"] = float(v)
            targets[f"target/swa/{k}"] = float(v)

assert targets
run = wandb.init(
    entity="rtv2",
    project="2026-09-09_pretrain",
    name="rt-plurel-target",
    id="rt-plurel-target",
    resume="allow",
)
for step in (0, 2**15):
    run.log({"step": step, **targets})
run.finish()
print(f"logged {len(targets)} rt-plurel target series")
