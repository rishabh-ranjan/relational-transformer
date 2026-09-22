import json
import math
from pathlib import Path

from expts.repaper.config import SHARE


def read_source(source_root: str) -> list[dict]:
    root = Path(source_root).expanduser()
    out = []
    for meta in sorted(root.glob("*/*_meta.json")):
        m = json.loads(meta.read_text())
        kept, sub = m["sampled_rows"], m["n_substituted"]
        req = kept + sub
        assert req > 0, f"{meta}: no requested rows"
        s = sub / req
        out.append(
            {
                "db": meta.parent.name,
                "task": m["task"],
                "n": int(m["total_nodes"]),
                "s": s,
                "labellable": m["total_nodes"] * (1.0 - s),
            }
        )
    return out


def solve_alpha(rec: list[dict], budget: int, floor: int) -> float:
    def total(a: float) -> float:
        return sum(
            min(r["labellable"], max(min(r["labellable"], floor), a * r["labellable"]))
            for r in rec
        )

    assert total(1.0) > budget, "budget exceeds the whole corpus; drop the plan"
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if total(mid) > budget:
            hi = mid
        else:
            lo = mid
    return lo


def build_plan(*, source_root: str, out_path: str, budget: int, floor: int) -> dict:
    rec = read_source(source_root)
    assert rec, f"no manifests under {source_root}"
    alpha = solve_alpha(rec, budget, floor)

    rows, target = {}, {}
    for r in rec:
        labellable = r["labellable"]
        t = min(labellable, max(min(labellable, floor), alpha * labellable))
        q = min(r["n"], math.ceil(t / max(1.0 - r["s"], 1e-9)))
        assert q >= min(r["n"], floor), f"{r['db']}/{r['task']} below the floor"
        key = f"{r['db']}/{r['task']}"
        rows[key] = int(q)
        target[key] = round(t)

    plan = {
        "budget": budget,
        "floor": floor,
        "alpha": alpha,
        "source_root": source_root,
        "n_tasks": len(rows),
        "sum_requested": sum(rows.values()),
        "sum_target_kept": sum(target.values()),
        "rows": rows,
        "target_kept": target,
    }
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1))
    return plan


plan = build_plan(
    source_root=f"{SHARE}/features_join_u12",
    out_path=f"{SHARE}/rows_plan_join_u12_prop.json",
    budget=200_000_000,
    floor=4096,
)
print(
    f"tasks {plan['n_tasks']:,}  alpha {plan['alpha']:.6f}  "
    f"requested {plan['sum_requested']:,}  target kept "
    f"{plan['sum_target_kept']:,}  "
    f"~{plan['sum_requested'] * 1024 / 2**30:.0f} GiB"
)
