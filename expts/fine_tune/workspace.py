"""Build the wandb workspace for this experiment: one panel per metric, each
holding the live curve, its SWA twin, and the published ceiling.

    pixi run python expts/fine_tune/workspace.py

wandb has no horizontal-reference-line primitive, so `rt.train` logs each
ceiling from `submit.CEILINGS` as a constant at every step -- a flat series.
That series is a metric key of its own (`{key}/ceiling`), and wandb's default
auto-panels would put it in a panel by itself; this script is what pairs it
with the curve it bounds.

It rewrites the saved view named below wholesale -- edit this script, not the
UI, or the next run of it drops your changes.
"""

import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws

from submit import CEILINGS, TASKS

ENTITY = "rtv2"
PROJECT = "2026-08-06-fine_tune"
# The saved view this script owns. A named view is addressable and idempotent:
# saving again overwrites it rather than piling up "Unsaved view" copies.
VIEW = "ceilings"


def panel(key: str) -> wr.LinePlot:
    """The metric, its SWA twin, and the ceiling, on one y-axis.

    The keys are given exactly (not as a regex prefix) so `auc/val/mean` does
    not swallow the per-task curves, and the ceiling is listed last so it
    draws on top of the curve it bounds.
    """
    return wr.LinePlot(
        title=key,
        x="step",
        y=[key, f"swa_{key}", f"{key}/ceiling"],
        smoothing_show_original=True,
    )


def main() -> None:
    # Only the tasks this sweep actually submits: a panel whose keys no run
    # logs renders empty.
    keys = [
        k for k in CEILINGS if any(k.endswith(f"/{db}/{task}") for db, task in TASKS)
    ]
    sections = []
    for split in ("val", "test"):
        panels = [panel(k) for k in sorted(keys) if f"/{split}/" in k]
        if panels:
            sections.append(
                ws.Section(
                    name=f"{split} vs published best",
                    panels=panels,
                    is_open=True,
                )
            )
    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=VIEW,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
