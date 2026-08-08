"""Build the wandb workspace for this experiment: a panel for every key the
project logs -- metrics and machine telemetry alike -- with each metric's SWA
twin and published target folded into the panel of the curve they belong to.

    pixi run python expts/fine_tune/workspace.py

wandb has no horizontal-reference-line primitive, so `rt.train` logs each
target from `submit.targets_for` as a constant at every step -- a flat series.
That series is a metric key of its own (`{key}/target`), and wandb's default
auto-panels would put it in a panel by itself; this script is what pairs it
with the curve it bounds. The same goes for the `swa_` twins.

The key list comes from the runs themselves (the summary of every run, plus
the system stream), so a metric added to `rt.train` shows up here as soon as
one run has logged it -- no edit to this file needed. Targets the sweep will
log but has not yet are folded in from `submit.targets_for`.

It rewrites the saved view named below wholesale -- edit this script, not the
UI, or the next run of it drops your changes.
"""

import wandb
import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import execute_graphql

from submit import TASKS, targets_for

ENTITY = "rtv2"
PROJECT = "2026-08-06-fine_tune"
# The saved view this script owns. A named view is addressable and idempotent:
# saving again overwrites it rather than piling up "Unsaved view" copies.
VIEW = "targets"

# wandb's own bookkeeping series. Panels for these say nothing about a run.
INTERNAL = {"_runtime", "_step", "_timestamp", "_wandb", "step"}


def swa_twins(key: str) -> list[str]:
    """The names an SWA twin of `key` could go by.

    `rt.train` prefixes the whole key (`swa_auc/val/mean`) in some places and
    only the leaf (`val/swa_clf`) in others; both spellings are offered and
    the caller keeps whichever the project actually logs.
    """
    head, sep, leaf = key.rpartition("/")
    return [f"swa_{key}", f"{head}{sep}swa_{leaf}"] if sep else [f"swa_{key}"]


def panel(key: str, keys: set[str], x: str) -> wr.LinePlot:
    """The metric, its SWA twin, and the target, on one y-axis.

    The keys are given exactly (not as a regex prefix) so `auc/val/mean` does
    not swallow the per-task curves, and the target is listed last so it
    draws on top of the curve it bounds.
    """
    y = [key]
    y += [t for t in swa_twins(key) if t in keys]
    if f"{key}/target" in keys:
        y.append(f"{key}/target")
    return wr.LinePlot(title=key, x=x, y=y, smoothing_show_original=True)


def project_keys() -> set[str]:
    """Every key logged by any run in the project, system stream included.

    A run's summary carries one entry per metric key it ever logged, so the
    union over runs is the project's key space; `systemMetrics` is the same
    thing for the telemetry stream, which the summary does not cover.
    """
    api = wandb.Api()
    keys: set[str] = set()
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        keys |= set(run.summary.keys())
        keys |= set(run.systemMetrics.keys())
    # Targets this sweep will log but no run has reached yet.
    keys |= {f"{k}/target" for db, task in TASKS for k in targets_for(db, task)}
    return keys - INTERNAL


def section(name: str, keys: list[str], all_keys: set[str], x: str) -> ws.Section:
    return ws.Section(
        name=name,
        panels=[panel(k, all_keys, x) for k in keys],
        is_open=True,
    )


VIEWS_QUERY = """
query Views($entityName: String, $name: String) {
    project(name: $name, entityName: $entityName) {
        allViews(viewType: "project-view") {
            edges { node { id name displayName updatedAt } }
        }
    }
}
"""


def existing_view(display_name: str) -> tuple[str, str]:
    """The (internal name, id) of the saved view titled `display_name`.

    A view's identity to wandb is its slug (`nw-4gxr4eybu76-v`) and id, not
    the title shown in the UI; `Workspace.save()` mints a fresh slug whenever
    it has none, so saving a freshly built `Workspace` piles up a new copy
    each run rather than replacing the last. Handing it back the slug of the
    view already carrying our title turns the save into an overwrite.
    """
    api = wandb.Api()
    resp = execute_graphql(api, VIEWS_QUERY, {"entityName": ENTITY, "name": PROJECT})
    nodes = [e["node"] for e in resp["project"]["allViews"]["edges"]]
    mine = sorted(
        (n for n in nodes if n["displayName"] == display_name),
        key=lambda n: n["updatedAt"],
    )
    return (mine[-1]["name"], mine[-1]["id"]) if mine else ("", "")


def main() -> None:
    keys = project_keys()
    # A key that is some other key's twin or target gets no panel of its own:
    # it already rides along in that key's panel.
    folded = {t for k in keys for t in swa_twins(k) if t in keys}
    folded |= {f"{k}/target" for k in keys if f"{k}/target" in keys}
    leaders = sorted(keys - folded)

    system = [k for k in leaders if k.startswith("system.")]
    metrics = [k for k in leaders if not k.startswith("system.")]

    sections = []
    shown: set[str] = set()
    for split in ("val", "test"):
        # The published-target comparisons lead: they are what the sweep is
        # for. Only keys that actually have a target belong here.
        targeted = [k for k in metrics if f"/{split}/" in k and f"{k}/target" in keys]
        if targeted:
            sections.append(
                section(f"{split} vs published best", targeted, keys, "step")
            )
            shown |= set(targeted)

    rest = [k for k in metrics if k not in shown]
    # Everything else the training loop logs, grouped by its top-level
    # namespace so a section is one family of curves.
    for ns in sorted({k.split("/")[0] for k in rest}):
        sections.append(
            section(ns, [k for k in rest if k.split("/")[0] == ns], keys, "step")
        )

    # Telemetry is sampled on a wall-clock timer, not on the training step, so
    # it is plotted against runtime; `step` would bunch it all at the origin.
    if system:
        gpu = [k for k in system if k.startswith("system.gpu.")]
        host = [k for k in system if not k.startswith("system.gpu.")]
        if gpu:
            sections.append(section("system: gpu", gpu, keys, "_runtime"))
        if host:
            sections.append(section("system: host", host, keys, "_runtime"))

    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=VIEW,
        sections=sections,
        settings=ws.WorkspaceSettings(x_axis="step"),
    )
    workspace._internal_name, workspace._internal_id = existing_view(VIEW)
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
