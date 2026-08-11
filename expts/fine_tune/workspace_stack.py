"""Build the saved view that compares bce against huber on rel-stack.

    pixi run python expts/fine_tune/workspace_stack.py

The project's own layout is [`workspace.py`](workspace.py)'s and stays as it
is: this writes a *saved* view beside it, reached from the view menu or from
the `?nw=` link this prints. The two arms differ in one argument
(`submit_stack.loss_fn="bce"` against `submit.loss_fn="huber"`), and everything
that makes them comparable -- the panels, the targets, the axis -- is
`workspace.build`'s, so this takes that workspace and narrows it:

- the runset draws only the four runs of the comparison and groups on
  `loss_fn`, the one argument the arms differ in, so a panel -- which is one
  task already -- carries one curve per arm and the legend names the loss;
- panels for the other 19 benchmark tasks come out, and so does the val split.
  The project seeds every benchmark task so one starting later has a panel
  waiting, which is right for the project view and wrong here: they bury the
  two being compared.
"""

import math

import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import (
    _internal_name_to_url_query_str,
    _url_query_str_to_internal_name,
)

import workspace

TITLE = "rel-stack: bce vs huber"

# The view's slug: `nw-{SLUG}-v` is its internal name, and `?nw={SLUG}` the
# link that opens it. `-v` is the shape of a *saved* view, which is what the
# view menu lists; `-w` is the personal workspace's shape instead, and a view
# saved under it is stored but never listed and never loads. Fixed rather than
# the random one `Workspace.save_as_new_view` generates, so rerunning this
# overwrites the view rather than adding another: the upsert keys on the name.
#
# Alphanumeric, like the random id `save_as_new_view` would have made: a slug
# with a `-` in it saves and can be fetched by name, but the project's view
# list -- which is what the view menu is -- drops it, and a view the app cannot
# list is one it will not open either. `-v` and `nw-` inside a slug are the
# same trap twice over, since both wrappers come off by substring rather than
# as affixes; the assert in `main` is what holds all of it.
SLUG = "relstackbcevshuber"

# The four runs of the comparison, by the `run_name` config `submit` and
# `submit_stack` set: `{db}/{task}-{arm}`, with a `-bce` suffix on the bce arm.
# Filtered on the config and not on the run's display name, which carries the
# slurm job id too and gains a fresh one every requeue.
RUNS = [
    f"rel-stack/{task}-trainval{suffix}"
    for task in ("user-badge", "user-engagement")
    for suffix in ("", "-bce")
]

TASKS = frozenset({"rel-stack/user-badge", "rel-stack/user-engagement"})


def keeps(title: str) -> bool:
    """Whether a panel titled `title` belongs in this view.

    A panel title is its key with the section's prefix off, so the task is its
    last two components when it names one at all. A panel that names no task --
    the train curves, the telemetry, the `mean` lines -- is about the run
    rather than the task and stays.
    """
    task = "/".join(title.split("/")[-2:])
    return task not in workspace.sizes() or task in TASKS


def main() -> None:
    w = workspace.build(workspace.ENTITY, "2026-08-10-fine_tune", "epoch")

    sections = []
    for s in w.sections:
        # The val split says what the run scored on data the trainval arm
        # trained on: not a comparison, and beside the test panels it reads as
        # one.
        if s.name == "dashboard: auroc/val":
            continue
        s.panels = [p for p in s.panels if keeps(p.title)]
        # The System section is panel-less by design and filled by the app;
        # every other empty one is a section this view has nothing for.
        if s.panels or s.name == workspace.SYSTEM:
            s.layout_settings.rows = max(1, math.ceil(len(s.panels) / workspace.COLS))
            sections.append(s)
    w.sections = sections

    w.runset_settings = ws.RunsetSettings(
        # By the loss and not by the run: a panel is one task already, so this
        # draws the two arms against each other and names them by what differs.
        groupby=[ws.Config("loss_fn")],
        filters=[ws.Config("run_name").isin(RUNS)],
    )

    # A saved view of its own rather than the personal workspace, which is
    # `workspace.py`'s to write. No id: the upsert in `workspace.save` keys on
    # the name, creating the view the first time and rewriting it after.
    name = _url_query_str_to_internal_name(SLUG)
    assert SLUG.isalnum() and _internal_name_to_url_query_str(name) == SLUG, SLUG
    w.name = TITLE
    w._internal_name = name
    w._internal_id = ""

    workspace.save(w)
    print(f"https://wandb.ai/{w.entity}/{w.project}?nw={SLUG}")


if __name__ == "__main__":
    main()
