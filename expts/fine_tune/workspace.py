"""Build the wandb workspace for this experiment: a panel for every key the
project logs -- metrics and machine telemetry alike -- with each metric's SWA
twin and published target folded into the panel of the curve they belong to.

    pixi run python expts/fine_tune/workspace.py [--project 2026-08-07-fine_tune]

wandb has no horizontal-reference-line primitive, so `rt.train` logs each
target from `submit.targets_for` as a constant at every step -- a flat series.
That series is a metric key of its own (`target/{key}`), and wandb's default
auto-panels would put it in a panel by itself; this script is what pairs it
with the curve it bounds. The same goes for the `swa/` twins.

The leading `dashboard:` sections are the hand-arranged ones -- one per metric
and split, sized so the whole set is on one page -- and every remaining key
gets a panel further down, grouped by namespace: nothing a run logs is dropped
just because this script did not anticipate it, telemetry included.

The key list comes from the runs themselves (the summary of every run, plus
the system stream), so this works against any project, and a metric added to
`rt.train` shows up as soon as one run has logged it. Targets the sweep will
log but has not yet are folded in from `submit.targets_for`.

It rewrites the saved view named below wholesale -- edit this script, not the
UI, or the next run of it drops your changes.
"""

import argparse
import math

import wandb
import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import execute_graphql

from submit import TASKS, targets_for

ENTITY = "rtv2"
PROJECT = "2026-08-07-fine_tune"
# The saved view this script owns. Saving reuses the view already carrying
# this title (see `existing_view`), so a rerun replaces it in place.
VIEW = "targets"

# The metric families that get a hand-arranged `dashboard:` section, in the
# order they should appear.
METRICS = ("auroc", "nmae")
SPLITS = ("val", "test")

# wandb's own bookkeeping series. Panels for these say nothing about a run.
INTERNAL = {"_runtime", "_step", "_timestamp", "_wandb", "step"}


def swa_key(key: str) -> str:
    """Where `rt.train` logs the SWA twin of `key`."""
    return f"swa/{key}"


def target_key(key: str) -> str:
    """Where `rt.train` logs the published target for `key`."""
    return f"target/{key}"


def panel(key: str, keys: set[str], x: str) -> wr.LinePlot:
    """The metric, its SWA twin, and the target, on one y-axis.

    The keys are given exactly (not as a regex prefix) so `auroc/val/mean`
    does not swallow the per-task curves, and the target is listed last so it
    draws on top of the curve it bounds.
    """
    y = [k for k in (key, swa_key(key), target_key(key)) if k in keys]
    return wr.LinePlot(title=key, x=x, y=y, smoothing_show_original=True)


def section(name: str, keys: list[str], all_keys: set[str], x: str) -> ws.Section:
    """One section, sized so every panel in it is on the first page.

    wandb paginates a section at columns x rows and defaults to 3x2, which
    hides most of a twelve-task sweep behind a pager; the grid here is the
    squarest one that holds the whole section.
    """
    cols = max(1, math.ceil(math.sqrt(len(keys))))
    return ws.Section(
        name=name,
        panels=[panel(k, all_keys, x) for k in keys],
        is_open=True,
        layout_settings=ws.SectionLayoutSettings(
            columns=cols,
            rows=math.ceil(len(keys) / cols),
        ),
    )


def project_keys(entity: str, project: str) -> set[str]:
    """Every key logged by any run in the project, system stream included.

    A run's summary carries one entry per metric key it ever logged, so the
    union over runs is the project's key space; `systemMetrics` is the same
    thing for the telemetry stream, which the summary does not cover.
    """
    api = wandb.Api()
    keys: set[str] = set()
    for run in api.runs(f"{entity}/{project}"):
        keys |= set(run.summary.keys())
        keys |= set(run.systemMetrics.keys())
    # The sweep's own metrics and their targets, whether or not a run has got
    # that far: a task still queueing at the first eval should have its panel
    # waiting for it, not appear halfway through the sweep. Both halves are
    # seeded -- the target alone would land in a panel of its own.
    for db, task in TASKS:
        for k in targets_for(db, task):
            keys |= {k, target_key(k)}
    return keys - INTERNAL


VIEWS_QUERY = """
query Views($entityName: String, $name: String) {
    project(name: $name, entityName: $entityName) {
        allViews(viewType: "project-view") {
            edges { node { id name displayName updatedAt } }
        }
    }
}
"""


def existing_view(entity: str, project: str, display_name: str) -> tuple[str, str]:
    """The (internal name, id) of the saved view titled `display_name`.

    A view's identity to wandb is its slug (`nw-4gxr4eybu76-v`) and id, not
    the title shown in the UI; `Workspace.save()` mints a fresh slug whenever
    it has none, so saving a freshly built `Workspace` piles up a new copy
    each run rather than replacing the last. Handing it back the slug of the
    view already carrying our title turns the save into an overwrite.
    """
    api = wandb.Api()
    resp = execute_graphql(api, VIEWS_QUERY, {"entityName": entity, "name": project})
    nodes = [e["node"] for e in resp["project"]["allViews"]["edges"]]
    mine = sorted(
        (n for n in nodes if n["displayName"] == display_name),
        key=lambda n: n["updatedAt"],
    )
    return (mine[-1]["name"], mine[-1]["id"]) if mine else ("", "")


def build(entity: str, project: str, view: str) -> ws.Workspace:
    keys = project_keys(entity, project)
    # A key that is some other key's twin or target gets no panel of its own:
    # it already rides along in that key's panel.
    folded = {swa_key(k) for k in keys} | {target_key(k) for k in keys}
    leaders = sorted(keys - folded)

    sections, shown = [], set()

    def dashboard(name: str, picked: list[str], x: str = "step") -> None:
        if picked:
            sections.append(section(f"dashboard: {name}", picked, keys, x))
            shown.update(picked)

    for metric in METRICS:
        for split in SPLITS:
            # The per-task curves against their published bests: what the
            # sweep is for. The `mean` keys are held back for their own
            # sections, which put the two metrics side by side.
            dashboard(
                f"{metric}/{split}",
                [
                    k
                    for k in leaders
                    if k.startswith(f"{metric}/{split}/") and not k.endswith("/mean")
                ],
            )
    for split in SPLITS:
        dashboard(
            f"mean/{split}",
            [k for k in leaders if k in {f"{m}/{split}/mean" for m in METRICS}],
        )

    # Everything else, so nothing a run logs goes missing from the view,
    # grouped by top-level namespace.
    rest = [k for k in leaders if k not in shown and not k.startswith("system.")]
    for ns in sorted({k.split("/")[0] for k in rest}):
        sections.append(
            section(ns, [k for k in rest if k.split("/")[0] == ns], keys, "step")
        )
    # Telemetry is the app's own: a section literally named "System" is filled
    # in at render time from the system stream (that is how the default
    # workspace shows those charts -- its saved spec holds an empty section of
    # that name). Panels we write ourselves against the `system.*` keys the
    # API reports render blank, so the section is left empty on purpose.
    if any(k.startswith("system.") for k in keys):
        sections.append(ws.Section(name="System", panels=[], is_open=False))

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=view,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace._internal_name, workspace._internal_id = existing_view(
        entity, project, view
    )
    return workspace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--view", default=VIEW)
    a = p.parse_args()

    workspace = build(a.entity, a.project, a.view)
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
