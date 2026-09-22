import json
import re
import time
from pathlib import Path


def main(
    *,
    ckpt_dir: str,
    features_root: str,
    tabpfn_dir: str,
    relbench_pre_dir: str,
    relbench_features_root: str,
    relbench_labels_root: str,
    relbench_n_ctx: int,
    relbench_n_query: int,
    seed: int,
    swa_momentum_per_step: float,
    ckpt_every: int,
    ema_snapshot_steps: list,
    probe_n_tasks: int,
    probe_n_ctx: int,
    probe_n_query: int,
    probe_min_rows: int,
    probe_seed: int,
    d_feat: int,
    out_json: str,
) -> None:
    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch

    from rt.eval.metrics import metric_for
    from rt.train.swa import SwaState

    from expts.repaper.adapter.train_adapter import (
        build_adapter,
        evaluate_relbench,
        load_relbench,
        loss_and_pred,
        make_ests,
    )

    device = "cuda"
    out = Path(out_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        Path(ckpt_dir).expanduser().glob("adapter_step*.pt"),
        key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)),
    )
    assert paths, f"no checkpoints under {ckpt_dir}"
    cks = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    steps = [int(c["step"]) for c in cks]
    ref = cks[0]
    d_out = int(ref["d_out"])
    print(f"{len(cks)} checkpoints, steps {steps[0]}..{steps[-1]}, d_out {d_out}", flush=True)

    # The Standardize buffers are supposed to be frozen and identical, which is
    # what makes averaging head.weight alone the whole of the average. Checked,
    # not assumed.
    buf_dev = {}
    for k in ("0.mean", "0.scale"):
        buf_dev[k] = max(
            float((c["state_dict"][k] - ref["state_dict"][k]).abs().max()) for c in cks
        )
    print(f"standardiser buffer max |delta| across checkpoints: {buf_dev}", flush=True)
    assert max(buf_dev.values()) == 0.0, buf_dev
    for c in cks:
        assert c["adapter_kind"] == ref["adapter_kind"]
        assert int(c["d_out"]) == d_out
        assert set(c["state_dict"]) == {"0.mean", "0.scale", "1.weight"}

    n = len(cks)
    W = torch.stack([c["state_dict"]["1.weight"].float() for c in cks])

    def sd_for(w):
        return {
            "0.mean": ref["state_dict"]["0.mean"].clone(),
            "0.scale": ref["state_dict"]["0.scale"].clone(),
            "1.weight": w.clone(),
        }

    # rt-j's own averaging scheme, rt.train.swa.SwaState, applied
    # retrospectively. _train.py:920 calls swa.update(master.items()) once per
    # optimiser step on the fp32 master *parameters* -- buffers are never
    # averaged, which is why only 1.weight moves here. SwaState.update is a
    # bias-corrected ema, alpha = (1 - m) / (1 - m**n) (swa.py:17), so the
    # state it is constructed from (swa.py:7, the init) is fully discarded by
    # the first update (alpha = 1 at n = 1) and the weight on update i is
    # proportional to m**(n - i).
    #
    # The checkpoints are ckpt_every steps apart, so one checkpoint-level
    # update has to stand in for ckpt_every per-step ones. Under this rule the
    # weight profile is geometric with ratio m per step, so ckpt_every steps
    # decay the old value by m**ckpt_every: the horizon-matched per-checkpoint
    # momentum is m**ckpt_every.
    m_ck = float(swa_momentum_per_step**ckpt_every)
    print(
        f"swa momentum {swa_momentum_per_step} per step, "
        f"{m_ck:.6f} per {ckpt_every}-step checkpoint; "
        f"horizon 1/(1-m) = {1 / (1 - swa_momentum_per_step):.0f} steps "
        f"= {1 / (1 - m_ck):.2f} checkpoints",
        flush=True,
    )

    assert steps[0] == 0 and all(
        b - a == ckpt_every for a, b in zip(steps, steps[1:])
    ), steps
    swa = SwaState([("1.weight", W[0])], momentum=m_ck)
    sets = []
    snaps = [s for s in ema_snapshot_steps if s <= steps[-1]]
    if steps[-1] not in snaps:
        snaps.append(steps[-1])
    for i in range(1, n):
        swa.update([("1.weight", W[i])])
        if steps[i] in snaps:
            sets.append(
                {
                    "label": f"ema_at_step{steps[i]}",
                    "kind": "ema",
                    "members": steps[1 : i + 1],
                    "step": steps[i],
                    "swa_n": swa.n,
                    "w": swa.params["1.weight"].clone(),
                }
            )
    sets.reverse()
    # Evaluated last-to-first so that, if the wall clock runs out, what is
    # missing is the early single checkpoints rather than the averages.
    for i in reversed(range(n)):
        sets.append(
            {
                "label": f"ckpt_step{steps[i]}",
                "kind": "single",
                "members": [steps[i]],
                "step": steps[i],
                "swa_n": None,
                "w": W[i],
            }
        )

    for s in sets:
        s["dist_from_step0"] = float((s["w"] - W[0]).norm())
        s["dist_from_final"] = float((s["w"] - W[-1]).norm())
        s["weight_norm"] = float(s["w"].norm())

    ests = make_ests(tabpfn_dir, device, seed, d_out, "fp32")
    adapter = build_adapter(
        ref["adapter_kind"], d_feat, d_out, ref["hidden_dim"], ref["stats_path"], device
    ).eval()

    relbench = load_relbench(
        relbench_pre_dir,
        relbench_features_root,
        relbench_labels_root,
        relbench_n_ctx,
        relbench_n_query,
        seed,
    )

    # The fixed training-distribution sample, built exactly as
    # eval_checkpoints.py builds it (same seed, same draw order), so the same
    # tasks and the same context/query rows are scored for every weight set.
    root = Path(features_root).expanduser()
    entries = []
    for mp in sorted(root.glob("*/*_meta.json")):
        m = json.loads(mp.read_text())
        if m["shape"][0] < probe_min_rows or (m["n_distinct_labels"] or 0) < 2:
            continue
        entries.append(
            {
                "db": mp.parent.name,
                "task": m["task"],
                "task_type": m["task_type"],
                "n": m["shape"][0],
                "bin": mp.with_name(mp.name.replace("_meta.json", "_target.bin")),
                "rows": mp.with_name(mp.name.replace("_meta.json", "_rows.npz")),
                "weight": float(min(m["total_nodes"], 100_000)),
            }
        )
    w = np.array([e["weight"] for e in entries])
    rng = np.random.default_rng(probe_seed)
    pick = rng.choice(len(entries), size=probe_n_tasks, replace=False, p=w / w.sum())
    probe = []
    for i in pick:
        e = entries[int(i)]
        n_c = min(probe_n_ctx, e["n"] - probe_n_query)
        if n_c < 64:
            continue
        idx = np.sort(rng.choice(e["n"], size=n_c + probe_n_query, replace=False))
        x = np.memmap(e["bin"], dtype=ml_dtypes.bfloat16, mode="r", shape=(e["n"], d_feat))
        emb = torch.from_numpy(np.ascontiguousarray(x[idx]).view(np.uint16)).view(
            torch.bfloat16
        )
        y = torch.from_numpy(np.load(e["rows"])["labels"][idx].astype("float32"))
        if e["task_type"] == "reg" and float(y[:n_c].std()) < 1e-6:
            continue
        probe.append({**e, "emb": emb, "y": y, "n_ctx": n_c})
    by = {t: sum(1 for p in probe if p["task_type"] == t) for t in ("clf", "reg")}
    print(f"probe set: {len(probe)} tasks {by}", flush=True)

    def score_one(task_type, emb, y, n_ctx):
        loss, pred, truth = loss_and_pred(ests, adapter, task_type, emb, y, n_ctx, device)
        if loss is None:
            return None, None
        try:
            _nm, v = metric_for(
                task_type, truth.float().cpu().numpy(), pred.float().cpu().numpy()
            )
        except ValueError:
            v = None
        return float(loss), (None if v is None else float(v))

    @torch.no_grad()
    def score_val():
        per = {}
        for t in relbench:
            ce, cy = t["ctx"]
            qe, qy = t["query"]
            lo, mv = score_one(
                t["task_type"], torch.cat([ce, qe]), torch.cat([cy, qy]), len(cy)
            )
            per[f"{t['db']}/{t['task']}"] = {
                "task_type": t["task_type"],
                "loss": lo,
                "metric": mv,
            }
        return per

    @torch.no_grad()
    def score_probe():
        per = {}
        for p in probe:
            lo, mv = score_one(p["task_type"], p["emb"], p["y"], p["n_ctx"])
            per[f"{p['db']}/{p['task']}"] = {
                "task_type": p["task_type"],
                "loss": lo,
                "metric": mv,
            }
        return per

    def agg(per):
        o = {}
        for tt in ("clf", "reg"):
            for what in ("loss", "metric"):
                vs = [
                    d[what]
                    for d in per.values()
                    if d["task_type"] == tt and d[what] is not None
                ]
                o[f"{tt}_{what}"] = float(np.mean(vs)) if vs else None
                o[f"{tt}_{what}_n"] = len(vs)
        return o

    rows = []
    t0 = time.time()
    checked_against_evaluate_relbench = None
    for si, s in enumerate(sets):
        adapter.load_state_dict(sd_for(s["w"]))
        val = score_val()
        prb = score_probe()
        if s["label"] == f"ckpt_step{steps[-1]}":
            # Cross-check: the val numbers here come from loss_and_pred, the
            # same call evaluate_relbench makes, but reported per task and with
            # the loss kept. Run the library function once on the same weights
            # and require the metrics to agree.
            rb = evaluate_relbench(relbench, ests, adapter, device)
            mine = agg(val)
            checked_against_evaluate_relbench = {
                "evaluate_relbench": {k: v[0] for k, v in rb.items()},
                "swa_eval": {"clf": mine["clf_metric"], "reg": mine["reg_metric"]},
            }
            print(f"cross-check: {checked_against_evaluate_relbench}", flush=True)
        va, pa = agg(val), agg(prb)
        rows.append(
            {
                "label": s["label"],
                "kind": s["kind"],
                "step": s["step"],
                "swa_n": s["swa_n"],
                "n_members": len(s["members"]),
                "members": s["members"],
                "weight_norm": s["weight_norm"],
                "dist_from_step0": s["dist_from_step0"],
                "dist_from_final": s["dist_from_final"],
                "val": {k: v for k, v in va.items()},
                "probe": {k: v for k, v in pa.items()},
                "val_per_task": val,
                "probe_per_task": prb,
            }
        )
        out.write_text(
            json.dumps(
                {
                    "steps": steps,
                    "swa_momentum_per_step": swa_momentum_per_step,
                    "swa_momentum_per_ckpt": m_ck,
                    "ckpt_every": ckpt_every,
                    "buffer_max_delta": buf_dev,
                    "cross_check": checked_against_evaluate_relbench,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(
            f"[{si + 1}/{len(sets)}] {s['label']:<24} "
            f"val loss clf {va['clf_loss']:.4f} reg {va['reg_loss']:.4f} "
            f"auroc {va['clf_metric']:.4f} nmae {va['reg_metric']:.4f} | "
            f"probe loss clf {pa['clf_loss']:.4f} reg {pa['reg_loss']:.4f} "
            f"auroc {pa['clf_metric']:.4f} nmae {pa['reg_metric']:.4f} "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    print(f"wrote {out}", flush=True)
