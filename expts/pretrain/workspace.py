import wandb_workspaces.workspaces as ws

from expts.fine_tune.workspace import (
    COLS,
    INTERNAL,
    SYSTEM,
    TRAIN_ORDER,
    logged_keys,
    panel,
    personal_view,
    save,
    section,
    step_vs_runtime,
    swa_key,
    target_key,
    task_size,
)
from expts.pretrain.submit_marlowe import args


def build(entity: str, project: str, targets: dict[str, float]) -> ws.Workspace:
    keys = set(logged_keys(entity, project)) - INTERNAL
    for k in targets:
        keys |= {k, target_key(k)}
    folded = {swa_key(k) for k in keys} | {target_key(k) for k in keys}
    leaders = sorted(keys - folded)

    sections, shown = [], set()

    def dashboard(name: str, picked: list[str]) -> None:
        if picked:
            sections.append(
                section(f"dashboard: {name}", picked, keys, "step", prefix=f"{name}/")
            )
            shown.update(picked)

    dashboard("val", [k for k in leaders if k.endswith("/val/mean")])
    for metric in ("auroc", "nmae"):
        picked = [
            k
            for k in leaders
            if k.startswith(f"{metric}/val/") and not k.endswith("/mean")
        ]
        picked.sort(key=task_size)
        dashboard(f"{metric}/val", picked)

    train = [
        panel(k, keys, "step", "train/") if k else step_vs_runtime()
        for k in TRAIN_ORDER
        if k is None or k in leaders
    ]
    if any(k in leaders for k in TRAIN_ORDER if k):
        sections.append(
            ws.Section(
                name="dashboard: train",
                panels=train,
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=COLS, rows=1),
            )
        )
        shown.update(k for k in TRAIN_ORDER if k)

    rest = [k for k in leaders if k not in shown and not k.startswith("system.")]
    for ns in sorted({k.split("/")[0] for k in rest}):
        sections.append(
            section(
                ns,
                [k for k in rest if k.split("/")[0] == ns],
                keys,
                "step",
                is_open=False,
            )
        )
    if any(k.startswith("system.") for k in keys):
        sections.append(ws.Section(name=SYSTEM, panels=[], is_open=False))

    name, id, display_name = personal_view(entity, project)
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=display_name,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis="step"),
        auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(groupby=[ws.Config("run_name")]),
    )
    workspace._internal_name, workspace._internal_id = name, id
    return workspace


a = args()
print(save(build(a["entity"], a["project"], a["targets"])))
# print(save(build(a["entity"], a["project"].replace("pretrain", "pretrain-abl"), a["targets"])))
