"""Build the saved view that compares bce against huber on rel-stack.

    pixi run python expts/fine_tune/workspace_stack.py

The project's own layout is [`workspace.py`](workspace.py)'s and stays as it
is: this writes a *saved* view beside it, reached from the view menu or from
the `?nw=` link this prints. The two arms differ in one argument
(`submit_stack.loss_fn="bce"` against `submit.loss_fn="huber"`), and everything
that makes them comparable -- the panels, the targets, the axis -- is
`workspace.build`'s, so this takes that workspace and narrows it:

- the runset draws only the four runs of the comparison and groups on
  `run_name`, so each panel carries one curve per arm per task and the legend
  names the arm;
- panels for the other 19 benchmark tasks come out. The project seeds all of
  them so a task starting later has a panel waiting, which is right for the
  project view and wrong here: 19 empty panels bury the two being compared.
"""

import math

import wandb_workspaces.workspaces as ws

import workspace

TITLE = "rel-stack: bce vs huber"

# The view's internal name, which is what a `?nw=` link carries. `nw-...-w` is
# the shape the app gives a workspace view -- the same shape as the personal
# one `workspace.personal_view` derives -- and a view named anything else is
# saved but never listed in the view menu. Fixed, so rerunning this overwrites
# the view rather than adding another: the upsert keys on the name.
SLUG = "nw-rel-stack-bce-vs-huber-w"

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
        s.panels = [p for p in s.panels if keeps(p.title)]
        # The System section is panel-less by design and filled by the app;
        # every other empty one is a section this view has nothing for.
        if s.panels or s.name == workspace.SYSTEM:
            s.layout_settings.rows = max(1, math.ceil(len(s.panels) / workspace.COLS))
            sections.append(s)
    w.sections = sections

    w.runset_settings = ws.RunsetSettings(
        groupby=[ws.Config("run_name")],
        filters=[ws.Config("run_name").isin(RUNS)],
    )

    # A saved view of its own rather than the personal workspace, which is
    # `workspace.py`'s to write. No id: the upsert in `workspace.save` keys on
    # the name, creating the view the first time and rewriting it after.
    w.name = TITLE
    w._internal_name = SLUG
    w._internal_id = ""

    workspace.save(w)
    # `?nw=` carries the view name without the wrapper the app puts around a
    # workspace view: `nw-<slug>-w` on disk is `?nw=<slug>` in the link, and the
    # full name in the link is a view the app never finds -- it spins instead of
    # loading.
    slug = SLUG.removeprefix("nw-").removesuffix("-w")
    print(f"https://wandb.ai/{w.entity}/{w.project}?nw={slug}")


if __name__ == "__main__":
    main()
