"""Build expts/fine_tune/results.md from results.csv.

Run from the repo root: `pixi run python expts/fine_tune/make_results.py`.
Rewrites results.md wholesale -- edit this script, not the markdown.

`my_results.py` imports `SHORT`, `NTRAIN`, `stds`, `table` and `legend` from
here to build the same tables with our own row in them, so the two documents
cannot drift in ordering, bolding or units.
"""

import pandas as pd
import json
from huggingface_hub import hf_hub_download

stds = json.load(
    open(
        hf_hub_download(
            "stanford-star/relbench", "regression_stds.json", repo_type="dataset"
        )
    )
)["stds"]

SHORT = {
    "rel-amazon/item-churn": "a/ichurn",
    "rel-amazon/user-churn": "a/uchurn",
    "rel-avito/user-clicks": "v/clicks",
    "rel-avito/user-visits": "v/visits",
    "rel-event/user-ignore": "e/ignore",
    "rel-event/user-repeat": "e/repeat",
    "rel-f1/driver-dnf": "f/dnf",
    "rel-f1/driver-top3": "f/top3",
    "rel-hm/user-churn": "h/churn",
    "rel-stack/user-badge": "s/badge",
    "rel-stack/user-engagement": "s/engage",
    "rel-trial/study-outcome": "t/outcm",
    "rel-amazon/item-ltv": "a/i-ltv",
    "rel-amazon/user-ltv": "a/u-ltv",
    "rel-avito/ad-ctr": "v/ad-ctr",
    "rel-event/user-attendance": "e/attend",
    "rel-f1/driver-position": "f/posit",
    "rel-hm/item-sales": "h/sales",
    "rel-stack/post-votes": "s/votes",
    "rel-trial/site-success": "t/succ",
    "rel-trial/study-adverse": "t/advrs",
}
assert max(map(len, SHORT.values())) <= 8 and len(set(SHORT.values())) == len(SHORT)

_t = pd.read_parquet(
    hf_hub_download(
        "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
    )
)
NTRAIN = {f"{r.database}/{r.task}": int(r.num_rows_train) for r in _t.itertuples()}


def human(n):
    for u, f in (("M", 1e6), ("k", 1e3)):
        if n >= f:
            return f"{n / f:.1f}".rstrip("0").rstrip(".") + u
    return str(n)


d = pd.read_csv("expts/fine_tune/results.csv")
d["pair"] = d.dataset + "/" + d.task
dflt = d[d.config_tag == "default"].copy()
dflt["run"] = "D"
hpo = d[d.selected].copy()
hpo["run"] = "H"
a = pd.concat([dflt, hpo])
a["row"] = a.model + " (" + a.run + ")"


def render(rows, nleft=2):
    def eff(c, i):
        return len(c) if (i < nleft or c.startswith("__")) else len(c) + 2

    w = [max(eff(r[i], i) for r in rows) for i in range(len(rows[0]))]

    def cell(c, i):
        if i < nleft:
            return c.ljust(w[i])
        return c.rjust(w[i]) if c.startswith("__") else c.rjust(w[i] - 2) + "  "

    out = ["| " + " | ".join(cell(c, i) for i, c in enumerate(rows[0])) + " |"]
    out.append(
        "|"
        + "|".join(
            ("-" * (w[i] + 1) + ":") if i >= nleft else (":" + "-" * (w[i] + 1))
            for i in range(len(w))
        )
        + "|"
    )
    out.append("| " + " | ".join(cell(c, i) for i, c in enumerate(rows[1])) + " |")
    out.append("| " + " | ".join(" " * w[i] for i in range(len(w))) + " |")
    for r in rows[2:]:
        out.append("| " + " | ".join(cell(c, i) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def table(sub, split, higher, mark=None):
    """The markdown table, and the task columns in the order it used them.

    `mark` is `{(row, pair): suffix}`, appended inside a cell after the number
    and outside the bold markers: a footnote on one value, not a value.
    """
    mark = mark or {}
    sub = sub.copy()
    if higher:
        sub["v"] = sub[f"{split}_score"] * 100
    else:
        sub["v"] = sub.apply(lambda r: r[f"{split}_score"] / stds[r.pair] * 100, axis=1)
    p = sub.pivot_table(index="row", columns="pair", values="v", aggfunc="first")
    p = p[sorted(p.columns, key=lambda c: NTRAIN[c])]
    p.insert(0, "mean", p.mean(axis=1))
    p = p.sort_values("mean", ascending=not higher)
    cols = list(p.columns)
    rank = p["mean"].rank(ascending=not higher, method="min").astype(int)
    best = {c: (p[c].max() if higher else p[c].min()) for c in cols}
    rows = [
        ["rank", "model", "mean"] + [SHORT[x] for x in cols[1:]],
        ["", "train size", ""] + [human(NTRAIN[x]) for x in cols[1:]],
    ]
    for name, row in p.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("-")
                continue
            s = f"{v:.1f}"
            s = f"__{s}__" if v == best[c] else s
            cells.append(s + mark.get((name, c), ""))
        rows.append([str(rank[name]), name] + cells)
    return render(rows), cols[1:]


def legend(pairs):
    rows = [["short", "dataset/task"]] + [[SHORT[p], p] for p in pairs]
    w = [max(len(r[i]) for r in rows) for i in range(2)]
    out = [
        "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(rows[0])) + " |",
        "|" + "|".join(":" + "-" * (w[i] + 1) for i in range(2)) + "|",
    ]
    out += [
        "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
        for r in rows[1:]
    ]
    return "\n".join(out)


def sections(frame, mark=None):
    """The four `## ...` sections, in the order results.md gives them."""
    clf = frame[frame.task_type == "BINARY_CLASSIFICATION"]
    reg = frame[frame.task_type == "REGRESSION"]
    secs = []
    for name, sub, split, higher in [
        ("Classification, val (AUROC %, higher is better)", clf, "val", True),
        ("Classification, test (AUROC %, higher is better)", clf, "test", True),
        ("Regression, val (nMAE %, lower is better)", reg, "val", False),
        ("Regression, test (nMAE %, lower is better)", reg, "test", False),
    ]:
        t, _ = table(sub, split, higher, mark)
        secs.append(f"## {name}\n\n{t}")
    return secs


NOTES = (
    "- `(D)` = default config; `(H)` = HPO, i.e. best of ~30 random-search trials by val score, refit and evaluated on test.\n"
    "- `(H)` val numbers are the selection criterion itself, so they are optimistically biased; `(D)` val numbers are not.\n"
    "- Task columns are ordered by train-set size (`num_rows_train` from `stanford-star/relbench`, `STATS/tasks.parquet`), smallest to largest; the `train size` row gives it.\n"
    "- `rank` = position by `mean` within that table (ties share the lower number).\n"
    "- `mean` = unweighted mean over the tasks in that table; rows sorted by it (best on top). Best value per column in bold.\n"
    "- nMAE = MAE / std(train target), std from `stanford-star/relbench` (`regression_stds.json`).\n"
)


def main():
    out = (
        "# Results\n\n"
        + "\n\n".join(sections(a))
        + "\n\n# Legend\n\n"
        + legend(list(SHORT))
        + "\n\n"
        + NOTES
    )
    open("expts/fine_tune/results.md", "w").write(out)
    print("wrote expts/fine_tune/results.md")


if __name__ == "__main__":
    main()
