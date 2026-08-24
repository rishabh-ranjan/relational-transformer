import argparse
import functools
import json
import math
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces.internal import execute_graphql

HERE = Path(__file__).parent

ENTITY = "rtv2"
PROJECT = "2026-08-08-fine_tune"

METRICS = ("auroc", "nmae")
SPLITS = ("val", "test")

BOX = 200

COLS = 8

SYSTEM = "System"

INTERNAL = {
    "_runtime",
    "_step",
    "_timestamp",
    "_wandb",
    "step",
    "epoch",
    "ens_size",
    "tune/idx",
}

LOG_Y = {"train/load_time", "train/sec_per_step"}

TRAIN_ORDER = (
    "train/load_time",
    "train/sec_per_step",
    None,
    "train/loss",
    "train/grad_norm",
    "train/lr",
)

PAGE_SIZE = 100


def published_best() -> dict[str, float]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stds = json.load(
        open(
            hf_hub_download(
                "stanford-star/relbench", "regression_stds.json", repo_type="dataset"
            )
        )
    )["stds"]

    raw = pd.read_csv(HERE / "results.csv")
    raw["pair"] = raw.dataset + "/" + raw.task
    dflt = raw[raw.config_tag == "default"].assign(arm="D")
    hpo = raw[raw.selected].assign(arm="H")
    d = pd.concat([dflt, hpo])
    d["row"] = d.model + " (" + d.arm + ")"

    out: dict[str, float] = {}
    for task_type, metric, higher in [
        ("BINARY_CLASSIFICATION", "auroc", True),
        ("REGRESSION", "nmae", False),
    ]:
        sub = d[d.task_type == task_type]
        best = max if higher else min
        for split in ("val", "test"):
            v = sub[f"{split}_score"] * 100
            if not higher:
                v = v / sub.pair.map(stds)
            for pair, x in v.groupby(sub.pair):
                out[f"{metric}/{split}/{pair}"] = float(best(x))
            out[f"{metric}/{split}/mean"] = float(best(v.groupby(sub.row).mean()))
    return out


def ntrain() -> dict[str, float]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": float(r.num_rows_train) for r in stats.itertuples()
    }


targets = functools.cache(published_best)
sizes = functools.cache(ntrain)


@functools.cache
def api() -> wandb.Api:
    return wandb.Api()


def swa_key(key: str) -> str:
    return f"swa/{key}"


def target_key(key: str) -> str:
    return f"target/{key}"


def best_key(key: str) -> str:
    return key.replace("tune/", "tune/best/", 1) if key.startswith("tune/") else ""


def axis(key: str, x: str) -> str:
    return "tune/idx" if key.startswith("tune/") else x


def task_size(key: str) -> float:
    return sizes().get("/".join(key.split("/")[2:]), math.inf)


def panel(key: str, keys: set[str], x: str, prefix: str = "") -> wr.LinePlot:
    y = [
        k
        for k in (key, swa_key(key), best_key(key), target_key(key))
        if k and k in keys
    ]
    return wr.LinePlot(
        title=key.removeprefix(prefix),
        x=axis(key, x),
        y=y,
        smoothing_show_original=True,
        log_y=key in LOG_Y or None,
    )


def step_vs_runtime() -> wr.LinePlot:
    return wr.LinePlot(
        title="step vs runtime", x="_runtime", y=["_step"], smoothing_show_original=True
    )


def section(
    name: str,
    keys: list[str],
    all_keys: set[str],
    x: str,
    is_open: bool = True,
    prefix: str = "",
) -> ws.Section:
    return ws.Section(
        name=name,
        panels=[panel(k, all_keys, x, prefix) for k in keys],
        is_open=is_open,
        layout_settings=ws.SectionLayoutSettings(
            columns=COLS,
            rows=max(1, math.ceil(len(keys) / COLS)),
        ),
    )


def logged_keys(entity: str, project: str) -> list[str]:
    keys: set[str] = set()
    cursor = None
    while True:
        page = execute_graphql(
            api(), KEYS_QUERY, {"entityName": entity, "name": project, "cursor": cursor}
        )["project"]["runs"]
        for edge in page["edges"]:
            for field in ("summaryMetrics", "systemMetrics"):
                if blob := edge["node"][field]:
                    keys |= set(json.loads(blob))
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return sorted(keys)


def project_keys(entity: str, project: str) -> set[str]:
    keys = set(logged_keys(entity, project))
    for k in targets():
        keys |= {k, target_key(k)}
    if any(k.startswith("tune/") for k in keys):
        for k in targets():
            if "/val/" in k and not k.endswith("/mean"):
                keys |= {f"tune/{k}", f"tune/best/{k}", target_key(f"tune/{k}")}
    return keys - INTERNAL


KEYS_QUERY = """
query Keys($entityName: String!, $name: String!, $cursor: String) {
    project(name: $name, entityName: $entityName) {
        runs(first: 500, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            edges { node { summaryMetrics systemMetrics } }
        }
    }
}
"""


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
    username = api().viewer.username
    slug = "nw-nwuser" + "".join(c for c in username if c.isalnum()) + "-w"
    resp = execute_graphql(api(), VIEWS_QUERY, {"entityName": entity, "name": project})
    nodes = [e["node"] for e in resp["project"]["allViews"]["edges"]]
    mine = [n for n in nodes if n["name"] == slug]
    if mine:
        return slug, mine[0]["id"], mine[0]["displayName"]
    return slug, "", f"{username.capitalize()}'s workspace"


def build(entity: str, project: str, x: str) -> ws.Workspace:
    keys = project_keys(entity, project)
    folded = (
        {swa_key(k) for k in keys}
        | {target_key(k) for k in keys}
        | {best_key(k) for k in keys if best_key(k)}
    )
    leaders = sorted(keys - folded)

    sections, shown = [], set()

    def dashboard(name: str, picked: list[str]) -> None:
        if picked:
            sections.append(
                section(f"dashboard: {name}", picked, keys, x, prefix=f"{name}/")
            )
            shown.update(picked)

    for metric in METRICS:
        for split in SPLITS:
            picked = [
                k
                for k in leaders
                if k.startswith(f"{metric}/{split}/") and not k.endswith("/mean")
            ]
            picked.sort(key=task_size)
            dashboard(f"{metric}/{split}", picked)
    tuned = set()
    for metric in METRICS:
        for split in SPLITS:
            picked = [
                k
                for k in leaders
                if k.startswith(f"tune/{metric}/{split}/") and not k.endswith("/mean")
            ]
            picked.sort(key=lambda k: task_size(k.removeprefix("tune/")))
            dashboard(f"tune/{metric}/{split}", picked)
            tuned.update(picked)
    dashboard("tune", [k for k in leaders if k.startswith("tune/") and k not in tuned])

    train = [
        panel(k, keys, x, "train/") if k else step_vs_runtime()
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
                x,
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
        settings=ws.WorkspaceSettings(x_axis=x),
        auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(groupby=[ws.Config("run_name")]),
    )
    workspace._internal_name, workspace._internal_id = name, id
    return workspace


def strip_max_runs(spec: object) -> None:
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
    view = workspace._to_model()
    view.spec.section.run_sets[0].run_feed.page_size = PAGE_SIZE
    spec = json.loads(view.spec.model_dump_json(by_alias=True, exclude_none=True))
    strip_max_runs(spec)
    for s in spec["section"]["panelBankConfig"]["sections"]:
        s.setdefault("flowConfig", {}).update(
            snapToColumns=False, boxWidth=BOX, boxHeight=BOX
        )
        if s["name"] == SYSTEM:
            s["isPanelsAuto"] = True
            s["defaultName"] = SYSTEM

    resp = execute_graphql(
        api(),
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
    return f"https://wandb.ai/{workspace.entity}/{workspace.project}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--x", default="epoch")
    a = p.parse_args()

    print(save(build(a.entity, a.project, a.x)))


if __name__ == "__main__":
    main()
