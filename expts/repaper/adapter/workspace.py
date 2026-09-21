import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws

from expts.fine_tune.workspace import (
    COLS,
    INTERNAL,
    SYSTEM,
    logged_keys,
    panel,
    personal_view,
    save,
    section,
    step_vs_runtime,
    target_key,
)
from expts.repaper.config import project

VAL = ["val/auroc", "val/nmae", "val/n_clf", "val/n_reg"]

TRAIN = [
    "train/loss/mean",
    "train/loss/clf",
    "train/loss/reg",
    "train/w_minus_i",
    "train/lr",
    "train/steps_per_sec",
    "train/minutes",
]


def per_task(title: str, regex: str) -> wr.LinePlot:
    return wr.LinePlot(
        title=title,
        x="step",
        metric_regex=regex,
        smoothing_show_original=True,
    )


def build(entity: str, name: str, targets: dict[str, float]) -> ws.Workspace:
    # Named, not discovered: the view is written before the first run exists,
    # and logged_keys() then has nothing to build a panel from.
    keys = (set(logged_keys(entity, name)) | set(VAL) | set(TRAIN)) - INTERNAL
    for k in targets:
        keys |= {k, target_key(k)}

    sections = [
        section("dashboard: relbench val", VAL, keys, "step"),
        # 21 tasks is 21 panels spelled out and 21 more whenever the task list
        # moves; one regex panel per metric follows whatever is logged.
        ws.Section(
            name="relbench val: per task",
            panels=[
                per_task("auroc per task", r"^val/auroc/"),
                per_task("nmae per task", r"^val/nmae/"),
            ],
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
        ),
        ws.Section(
            name="dashboard: train",
            panels=[panel(k, keys, "step", "train/") for k in TRAIN]
            + [step_vs_runtime()],
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=COLS, rows=1),
        ),
    ]

    shown = set(VAL) | set(TRAIN)
    rest = [
        k
        for k in sorted(keys)
        if k not in shown
        and not k.startswith(("val/", "target/", "system."))
    ]
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

    view_name, id, display_name = personal_view(entity, name)
    workspace = ws.Workspace(
        entity=entity,
        project=name,
        name=display_name,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis="step"),
        auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(groupby=[ws.Config("run_name")]),
    )
    workspace._internal_name, workspace._internal_id = view_name, id
    return workspace


print(
    save(
        build(
            "rtv2",
            # "vedanga-stanford-university",
            project("adapter"),
            # rt-j without an adapter, on the same 21 val tasks: the line every
            # val panel is read against (submit_train_adapter.py logs it).
            {"val/auroc": 0.7173, "val/nmae": 0.3584},
        )
    )
)
