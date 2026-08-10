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

It writes your personal workspace -- the view the project's URL opens on, so
the layout is there without picking anything from the view menu -- and writes
it wholesale: edit this script, not the UI, or the next run drops your changes.

New runs show up in the view on their own: every run in the project is in its
runset (no filters), no run limit is written (a panel draws the whole sweep,
and without the "Limited to N runs" caption any `max_runs` value earns), and
the run feed's page size is lifted off 10 so the run list is not stuck on
1-10. Panels are not
automatic --
`auto_generate_panels` is off, so a key this script has not seen gets its panel
by rerunning the script.
"""

import argparse
import json
import math

import wandb
import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import execute_graphql

from submit import TASKS, targets_for

ENTITY = "rtv2"
PROJECT = "2026-08-07-fine_tune"

# The metric families that get a hand-arranged `dashboard:` section, in the
# order they should appear.
METRICS = ("auroc", "nmae")
SPLITS = ("val", "test")

# Panels per row in every section -- an eighth of the width each, so a whole
# split's tasks read as a single strip rather than a block to scan.
COLS = 8

# The panel box, in pixels, square. wandb's own default is 460x300 at three
# columns, so a column is ~460px wide on the workspace the app sizes for;
# an eighth of that same width is what BOX is, and the height matches.
BOX = 3 * 460 // COLS

# The app-managed telemetry section. Its charts are generated client-side, so
# the saved spec carries the name and nothing else.
SYSTEM = "System"

# wandb's own bookkeeping series. Panels for these say nothing about a run.
INTERNAL = {"_runtime", "_step", "_timestamp", "_wandb", "step"}

# Keys whose panel gets a log y-axis: timings whose interesting structure is
# the occasional order-of-magnitude spike, which a linear axis flattens the
# rest of the curve to draw.
LOG_Y = {"train/load_time", "train/sec_per_step"}

# The `dashboard: train` section, in the order the panels should read. `None`
# is the step-vs-runtime panel, which is drawn from wandb's own bookkeeping
# series rather than from a logged key.
TRAIN_ORDER = (
    "train/load_time",
    "train/sec_per_step",
    None,
    "train/loss",
    "train/grad_norm",
    "train/lr",
)

# How many runs the run list shows before paging. Also 10 by default, which is
# what opens the list on "1-10"; 100 is what the app allows (it clamps anything
# larger down to this).
PAGE_SIZE = 100


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
    return wr.LinePlot(
        title=key, x=x, y=y, smoothing_show_original=True, log_y=key in LOG_Y or None
    )


def step_vs_runtime() -> wr.LinePlot:
    """How fast a run is actually moving: step against wall-clock seconds, so
    a run that has stalled or is crawling reads off the slope.

    `_step` and `_runtime` are wandb's own bookkeeping series -- excluded from
    the key space, which is why this panel is written by hand.
    """
    return wr.LinePlot(
        title="step vs runtime", x="_runtime", y=["_step"], smoothing_show_original=True
    )


def section(name: str, keys: list[str], all_keys: set[str], x: str) -> ws.Section:
    """One section, COLS panels wide and deep enough to hold all of them.

    wandb paginates a section at columns x rows and defaults to 3x2, which
    hides most of a twelve-task sweep behind a pager; the grid here never
    pages, and its width is fixed so panels are the same size section to
    section.
    """
    return ws.Section(
        name=name,
        panels=[panel(k, all_keys, x) for k in keys],
        is_open=True,
        layout_settings=ws.SectionLayoutSettings(
            columns=COLS,
            rows=max(1, math.ceil(len(keys) / COLS)),
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


def personal_view(entity: str, project: str) -> tuple[str, str, str]:
    """The (internal name, id, title) of the viewer's own workspace view.

    Opening a project lands on the personal workspace, so that is the view this
    script writes -- a saved view of its own would have to be picked from the
    view menu every time, and there is no "make this one the default" anywhere
    in the API. Its slug is derived from the username rather than random, which
    is what lets us address it before it exists (a project never visited in the
    UI has no such view yet, and upserting the slug creates it).
    """
    api = wandb.Api()
    username = api.viewer.username
    slug = "nw-nwuser" + "".join(c for c in username if c.isalnum()) + "-w"
    resp = execute_graphql(api, VIEWS_QUERY, {"entityName": entity, "name": project})
    nodes = [e["node"] for e in resp["project"]["allViews"]["edges"]]
    mine = [n for n in nodes if n["name"] == slug]
    if mine:
        return slug, mine[0]["id"], mine[0]["displayName"]
    return slug, "", f"{username.capitalize()}'s workspace"


def build(entity: str, project: str) -> ws.Workspace:
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
            # sweep is for. The `mean` keys are left out -- they are not a
            # task, and they fall through to the namespace grouping below.
            dashboard(
                f"{metric}/{split}",
                [
                    k
                    for k in leaders
                    if k.startswith(f"{metric}/{split}/") and not k.endswith("/mean")
                ],
            )
    # The `mean` keys get no dashboard of their own: they fall through to the
    # namespace grouping below with the rest of their metric.

    # The training curves, in reading order: how fast the run is moving first
    # (the two timings and the step-vs-runtime slope), then what it is doing
    # (loss, grad norm, lr). Hand-ordered because the namespace grouping below
    # sorts alphabetically, which interleaves the two halves.
    train = [
        panel(k, keys, "step") if k else step_vs_runtime()
        for k in TRAIN_ORDER
        if k is None or k in leaders
    ]
    sections.append(
        ws.Section(
            name="dashboard: train",
            panels=train,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=COLS, rows=1),
        )
    )
    shown.update(k for k in TRAIN_ORDER if k)

    # Everything else, so nothing a run logs goes missing from the view,
    # grouped by top-level namespace.
    rest = [k for k in leaders if k not in shown and not k.startswith("system.")]
    for ns in sorted({k.split("/")[0] for k in rest}):
        sections.append(
            section(ns, [k for k in rest if k.split("/")[0] == ns], keys, "step")
        )
    # Telemetry is the app's own: the default workspace holds a *panel-less*
    # section named "System" that the app fills at render time. `save` is what
    # marks it auto (see there); an empty section alone renders empty, and
    # panels written by hand against the `system.*` keys the API reports render
    # blank too.
    if any(k.startswith("system.") for k in keys):
        sections.append(ws.Section(name=SYSTEM, panels=[], is_open=False))

    name, id, display_name = personal_view(entity, project)
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=display_name,
        sections=sections,
        # No `max_runs` here, and `strip_max_runs` takes out the default the
        # SDK writes in its place -- see there for why any value is the wrong
        # one.
        settings=ws.WorkspaceSettings(x_axis="step"),
        # The sections here are the whole view: the app is not to append panels
        # of its own for keys logged after this script ran. A new key gets its
        # panel by rerunning the script, which folds it into the section it
        # belongs to rather than into an auto-generated one.
        auto_generate_panels=False,
    )
    workspace._internal_name, workspace._internal_id = name, id
    return workspace


def strip_max_runs(spec: object) -> None:
    """Drop every `maxRuns` from the spec, wherever the SDK wrote one.

    Leaving `WorkspaceSettings.max_runs` unset is not enough: the SDK fills it
    with wandb's own default of 10, and *any* value there is a run limit as far
    as the app is concerned -- it truncates the sweep and captions every panel
    "Limited to N runs for each visualized metric". Absent from the spec, the
    app draws the whole runset with no caption.
    """
    if isinstance(spec, dict):
        spec.pop("maxRuns", None)
        for v in spec.values():
            strip_max_runs(v)
    elif isinstance(spec, list):
        for v in spec:
            strip_max_runs(v)


UPSERT = """
mutation Upsert($id: ID, $entityName: String, $projectName: String, $name: String,
                $displayName: String, $type: String, $spec: String) {
    upsertView(input: {id: $id, entityName: $entityName, projectName: $projectName,
                       name: $name, displayName: $displayName, type: $type,
                       spec: $spec, createdUsing: WANDB_SDK}) {
        view { id name }
    }
}
"""


def save(workspace: ws.Workspace) -> str:
    """Save the workspace, with the System section marked auto.

    `wandb_workspaces` has no field for `isPanelsAuto`, the flag that tells the
    app to generate a section's panels itself -- which is the only way the
    system charts appear, since nothing in the run's logged keys draws them.
    So the spec is serialized, the flag set on that one section, and the
    result upserted directly rather than through `Workspace.save()`.

    The run feed's page size gets the same treatment, for the same reason: no
    field for it either, and its default of 10 is what makes the run list open
    on "1-10" with the rest of the sweep behind a pager.
    """
    view = workspace._to_model()
    view.spec.section.run_sets[0].run_feed.page_size = PAGE_SIZE
    spec = json.loads(view.spec.model_dump_json(by_alias=True, exclude_none=True))
    strip_max_runs(spec)
    for s in spec["section"]["panelBankConfig"]["sections"]:
        # Square panels. `SectionLayoutSettings` has no height field -- it
        # writes only the column and row counts -- so the box is sized here.
        # With `snapToColumns` on, the app takes the width from the column
        # count and the height from `boxHeight` verbatim; BOX is the width a
        # column comes out to at COLS across a full-width workspace, so
        # setting both to it is what makes the panel as tall as it is wide.
        s.setdefault("flowConfig", {}).update(boxWidth=BOX, boxHeight=BOX)
        if s["name"] == SYSTEM:
            s["isPanelsAuto"] = True
            s["defaultName"] = SYSTEM

    resp = execute_graphql(
        wandb.Api(),
        UPSERT,
        {
            "id": view.id or None,
            "entityName": view.entity,
            "projectName": view.project,
            "name": view.name,
            "displayName": view.display_name,
            "type": "project-view",
            "spec": json.dumps(spec),
        },
    )
    workspace._internal_name = resp["upsertView"]["view"]["name"]
    workspace._internal_id = resp["upsertView"]["view"]["id"]
    # Not `workspace.url`: that builds a `?nw=` link to a saved view, and the
    # personal workspace is what the bare project URL opens.
    return f"https://wandb.ai/{workspace.entity}/{workspace.project}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    a = p.parse_args()

    print(save(build(a.entity, a.project)))


if __name__ == "__main__":
    main()
