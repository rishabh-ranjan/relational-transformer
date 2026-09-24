import json
import time
from pathlib import Path

import numpy as np
import torch

CHUNK = 1024
VARIANTS = ("head", "swa")


def blob_dir(features_root, db, table):
    return Path(features_root).expanduser() / db / "pool_features" / table


def query_rows(pre_dir, db, table, context_rows, context_seed):
    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )
    from expts.repaper.baselines.rel2tabv2.context_sampler import UniformRandomSampler
    from expts.repaper.baselines.rel2tabv2.global_retriever import GlobalContextRetriever

    retr = GlobalContextRetriever(
        sampler=UniformRandomSampler(seed=context_seed),
        db=db,
        table=table,
        pre_dir=pre_dir,
        context_split="train",
        n_rows_list=[context_rows],
        ctx_keys=[0],
    )
    ctx = retr.context_node_idxs()[0]
    val = get_table_splits(load_table_info(pre_dir, db), table)["val"]
    min_offset, _total = table_offset_and_len(pre_dir, db, table)
    rest = np.arange(val["node_idx_offset"], val["node_idx_offset"] + val["num_nodes"])
    rest = np.setdiff1d(np.union1d(rest, [min_offset]), ctx)
    return ctx, rest


def featurize_task(*, db, table, pre_dir, ckpt, pool_ckpt, features_root, context_rows, context_seed, compile):
    from expts.repaper.adapter.pool_head import PoolHead
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset
    from rt.data import get_tasks, process_batch
    from rt.model import load_rt_model

    out = blob_dir(features_root, db, table)
    if (out / "meta.json").is_file():
        meta = json.loads((out / "meta.json").read_text())
        want = {"pool_ckpt": pool_ckpt, "rtj_ckpt": ckpt, "context_rows": context_rows, "context_seed": context_seed}
        assert {k: meta[k] for k in want} == want, f"{out} was featurized for {meta}, not {want}"
        print(f"{out} exists; not refeaturizing", flush=True)
        return
    device = "cuda"
    t0 = time.perf_counter()
    ctx, rest = query_rows(pre_dir, db, table, context_rows, context_seed)
    nodes = np.concatenate([ctx, rest])

    net, config = load_rt_model(ckpt, device=device, compile=compile)
    net = net.to(torch.bfloat16).eval()
    ck = torch.load(Path(pool_ckpt).expanduser(), map_location="cpu", weights_only=True)
    norm = ck.get("swiglu_norm", "none")
    heads = {}
    for v, key in zip(VARIANTS, ("state_dict", "swa_state_dict")):
        h = PoolHead(net.d_model, ck["n_queries"], norm).to(device)
        h.load_state_dict(ck[key], strict=True)
        heads[v] = h.eval()
    (task,) = get_tasks(pre_dir, [(db, table)], ("test",))
    ds = make_dataset([task], pre_dir, config, mmap_populate=True)
    print(f"{db}/{table}: {len(ctx)} context + {len(rest)} other rows, step {ck['step']}, "
          f"swiglu_norm {norm}, setup {time.perf_counter() - t0:.0f}s", flush=True)

    pooled = {v: torch.empty(len(nodes), ck["n_queries"], device=device) for v in VARIANTS}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(nodes), CHUNK):
            chunk = nodes[start : start + CHUNK]
            n_real = len(chunk)
            if n_real < CHUNK:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], CHUNK - n_real)])
            tup = ds.sampler.batch_for_nodes_py([int(v) for v in chunk], 0, LOCAL_CTX)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = net(batch, return_embeddings=True)
            keys = batch["col_name_idxs"].masked_fill(
                batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
            )
            si = keys.argsort(dim=-1, stable=True)
            is_t = batch["is_targets"].gather(1, si)
            assert (is_t.sum(1) == 1).all(), f"{db}/{table}: not one target per row"
            got = (batch["node_idxs"].gather(1, si) * is_t).sum(1)
            want = torch.as_tensor(chunk, device=device, dtype=got.dtype)
            assert bool((got == want).all()), f"{db}/{table}: sampler substituted a row near {start}"
            pad = batch["is_padding"].gather(1, si)
            for v, h in heads.items():
                pooled[v][start : start + n_real] = h.pool(x[:n_real], pad[:n_real])
            if (start // CHUNK) % 100 == 0:
                done = start + n_real
                print(f"  {done}/{len(nodes)} rows, {done / (time.perf_counter() - t0):.0f} rows/s", flush=True)
        feats = {v: heads[v].mix(pooled[v], len(ctx)).float().cpu().numpy() for v in VARIANTS}
    for v, f in feats.items():
        assert np.isfinite(f).all(), f"{db}/{table} {v}: non-finite features"
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(nodes, kind="stable")
    np.save(out / "node_idxs.npy", nodes[order].astype(np.int64))
    for v, f in feats.items():
        np.save(out / f"{v}.npy", f[order])
    (out / "meta.json").write_text(json.dumps({
        "pool_ckpt": pool_ckpt, "step": ck["step"], "swiglu_norm": norm, "rtj_ckpt": ckpt,
        "compile": compile, "n_context": len(ctx), "n_rows": len(nodes), "n_features": ck["n_queries"],
        "context_rows": context_rows, "context_seed": context_seed,
        "seconds": time.perf_counter() - t0,
    }, indent=1))
    print(f"{db}/{table}: wrote {len(nodes)} rows x {ck['n_queries']} to {out} "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    del net, heads, pooled
    torch.cuda.empty_cache()


class PoolFeaturizer:
    def __init__(self, features_root, db_tables, variant):
        assert variant in VARIANTS, variant
        self._features = {}
        for db, table in sorted(set(db_tables)):
            d = blob_dir(features_root, db, table)
            meta = json.loads((d / "meta.json").read_text())
            idx = np.load(d / "node_idxs.npy")
            vec = torch.from_numpy(np.load(d / f"{variant}.npy"))
            self._features[db, table] = (idx, vec)
            print(f"PoolFeaturizer[{variant}, step {meta['step']}]: {db}/{table} "
                  f"({len(idx)} rows, {vec.shape[1]} features)", flush=True)

    def compute_features(self, task, node_idxs, device):
        idx, vec = self._features[task.db_name, task.table_name]
        want = node_idxs.cpu().numpy()
        pos = np.searchsorted(idx, want).clip(max=len(idx) - 1)
        assert (idx[pos] == want).all(), (
            f"{task.db_name}/{task.table_name}: {int((idx[pos] != want).sum())} rows not featurized"
        )
        return vec[torch.from_numpy(pos)].to(device)


def main(*, featurize: dict, run: dict) -> None:
    from expts.repaper.baselines.rel2tabv2.run import main as run_main

    featurize_task(**featurize)
    run_main(**run)
