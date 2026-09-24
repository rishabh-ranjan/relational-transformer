import json
import time
import zlib
from pathlib import Path

CHUNK = 256
CELL_KEYS = ("node_idxs", "table_name_idxs", "col_name_idxs", "bfs_depths", "is_targets", "is_task_nodes", "sem_types", "is_padding")


def head_internals(head, x, pad):
    import math

    z = (x.float() - head.mean) / head.scale
    qk = head.wk.weight.t() @ head.q.t()
    logits = (z @ qk) / math.sqrt(head.d_model)
    logits = logits.masked_fill(pad[..., None], float("-inf"))
    attn = logits.softmax(dim=1)
    h = (attn * head.wv(z)).sum(dim=1)
    return attn, h


def swiglu_internals(head, h, n_ctx):
    from torch.nn import functional as F

    u = head.norm(h, n_ctx) if head.swiglu_norm == "context" else head.norm(h)
    pre = head.w1(u)
    gate = F.silu(pre)
    value = head.w3(u)
    gated = gate * value
    out = head.w2(gated)
    return {"norm_in": u, "gate_pre": pre, "gate": gate, "value": value, "gated": gated, "swiglu_out": out, "final": h + out}


def attn_aggregate(attn, cells, target_node, names, n_depth):
    import torch

    real = ~cells["is_padding"]
    is_t = cells["is_targets"].bool()
    seed_row = (cells["node_idxs"] == target_node[:, None]) & real & ~is_t
    other = real & ~is_t & ~seed_row
    ent = -(attn.clamp_min(1e-30).log() * attn).sum(1)
    agg = {
        "target": (attn * is_t[..., None]).sum(1).mean(0),
        "seed_row_other": (attn * seed_row[..., None]).sum(1).mean(0),
        "other_rows": (attn * other[..., None]).sum(1).mean(0),
        "entropy": ent.mean(0),
        "eff_cells": ent.exp().mean(0),
        "n_real_cells": real.sum(1).float().mean(),
        "max_weight": attn.amax(1).mean(0),
    }
    depth = cells["bfs_depths"].clamp(max=n_depth - 1)
    agg["by_depth"] = torch.stack([(attn * ((depth == d) & real)[..., None]).sum(1).mean(0) for d in range(n_depth)])
    by = {}
    for kind, key in (("table", "table_name_idxs"), ("column", "col_name_idxs")):
        idx = cells[key].masked_fill(~real, -1)
        out = {}
        for v in idx.unique().tolist():
            if v < 0:
                continue
            m = idx == v
            out[names[v]] = {
                "mass": (attn * m[..., None]).sum(1).mean(0),
                "cell_share": (m.sum(1).float() / real.sum(1).float()).mean(),
            }
        by[kind] = out
    return agg, by


def main(
    *,
    db: str,
    table: str,
    pre_dir: str,
    ckpt: str,
    pool_ckpt: str,
    out_dir: str,
    n_rows: int,
    n_samples: int,
    n_depth: int,
    seed: int,
    compile: bool,
) -> None:
    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import PoolHead
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset
    from expts.repaper.baselines.rel2tab.featurizer import get_table_splits, load_table_info
    from rt.data import get_tasks, process_batch
    from rt.model import load_rt_model

    device = "cuda"
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    name = f"{db}__{table}"
    if (out / f"{name}.json").is_file():
        print(f"{name} done; nothing to do", flush=True)
        return
    t0 = time.perf_counter()
    net, config = load_rt_model(ckpt, device=device, compile=compile)
    net = net.to(torch.bfloat16).eval()
    ck = torch.load(Path(pool_ckpt).expanduser(), map_location="cpu", weights_only=True)
    norm = ck.get("swiglu_norm", "none")
    head = PoolHead(net.d_model, ck["n_queries"], norm).to(device)
    head.load_state_dict(ck["state_dict"], strict=True)
    head.eval()

    (task,) = get_tasks(pre_dir, [(db, table)], ("test",))
    names = json.loads((Path(pre_dir).expanduser() / db / "text.json").read_text())
    ds = make_dataset([task], pre_dir, config, mmap_populate=False)
    sp = get_table_splits(load_table_info(pre_dir, db), table)["train"]
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    nodes = sp["node_idx_offset"] + np.sort(rng.choice(sp["num_nodes"], size=min(n_rows, sp["num_nodes"]), replace=False))
    xs, cells_all = [], {k: [] for k in CELL_KEYS}
    with torch.inference_mode():
        for start in range(0, len(nodes), CHUNK):
            chunk = nodes[start : start + CHUNK]
            tup = ds.sampler.batch_for_nodes_py([int(v) for v in chunk], 0, LOCAL_CTX)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = net(batch, return_embeddings=True)
            keys = batch["col_name_idxs"].masked_fill(batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max)
            si = keys.argsort(dim=-1, stable=True)
            cells = {k: batch[k].gather(1, si) for k in CELL_KEYS}
            is_t = cells["is_targets"].bool()
            assert (is_t.sum(1) == 1).all(), name
            got = (cells["node_idxs"] * is_t).sum(1)
            assert bool((got == torch.as_tensor(chunk, device=device, dtype=got.dtype)).all()), f"{name}: substituted row near {start}"
            xs.append(x)
            for k in CELL_KEYS:
                cells_all[k].append(cells[k])
        x = torch.cat(xs)
        cells = {k: torch.cat(v) for k, v in cells_all.items()}
        pad = cells["is_padding"]
        is_t = cells["is_targets"].bool()
        target_node = (cells["node_idxs"] * is_t).sum(1)
        attn, h = head_internals(head, x, pad)
        feats = {"rtj_target": x[is_t].float(), "attn_out": h}
        feats.update(swiglu_internals(head, h, len(nodes)))
        pairs = torch.stack([cells["table_name_idxs"][~pad], cells["col_name_idxs"][~pad]], 1).unique(dim=0).tolist()
        bad = [(names[t], names[c]) for t, c in pairs if not names[c].endswith(f" of {names[t]}")]
        assert not bad, f"{name}: column names do not match tables: {bad[:5]}"
        agg, by = attn_aggregate(attn, cells, target_node, names, n_depth)
        pick = np.sort(rng.choice(len(nodes), size=min(n_samples, len(nodes)), replace=False))
        sample = {
            "rows": nodes[pick],
            "attn": attn[pick].half().cpu().numpy(),
            **{f"cell_{k}": cells[k][pick].cpu().numpy() for k in CELL_KEYS},
        }
        qk = head.wk.weight.t() @ head.q.t()
        params = {k: v.float().cpu().numpy() for k, v in head.state_dict().items()}

    for k, v in feats.items():
        assert torch.isfinite(v).all(), f"{name} {k}: non-finite"
    np.savez(out / f"{name}_feats.npz", nodes=nodes, **{k: v.cpu().numpy() for k, v in feats.items()})
    np.savez(out / f"{name}_attn_samples.npz", **sample)
    np.savez(out / f"{name}_head_params.npz", qk=qk.float().cpu().numpy(), **params)
    tolist = lambda v: v.float().cpu().numpy().tolist()
    summary = {
        "task": f"{db}/{table}",
        "task_type": task.task_type,
        "n_rows": len(nodes),
        "pool_ckpt": pool_ckpt,
        "step": ck["step"],
        "swiglu_norm": norm,
        "attn": {k: tolist(v) for k, v in agg.items()},
        "attn_by": {kind: {n: {k: tolist(v) for k, v in d.items()} for n, d in grp.items()} for kind, grp in by.items()},
        "names": {str(i): names[i] for i in sorted({int(v) for k in ("table_name_idxs", "col_name_idxs") for v in cells[k][~pad].unique().tolist()})},
        "seconds": time.perf_counter() - t0,
    }
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(
        f"{name}: {len(nodes)} rows, {float(agg['n_real_cells']):.0f} cells/row, "
        f"target mass {float(agg['target'].mean()):.3f}, seed row {float(agg['seed_row_other'].mean()):.3f}, "
        f"eff cells {float(agg['eff_cells'].mean()):.1f}, {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
