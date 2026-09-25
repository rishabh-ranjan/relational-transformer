import json
import time
from pathlib import Path

import numpy as np
import torch

CELLS_PER_BATCH = 2**18


def blob_dir(features_root, db, table):
    return Path(features_root).expanduser() / db / "rtj_rows" / table


def make_dataset(task, pre_dir, config, local_ctx, bfs_width):
    from rt.data import RustlerDataset

    return RustlerDataset(
        tasks=[task],
        pre_dir=pre_dir,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[local_ctx],
        bfs_width_list=[bfs_width],
        num_walks=0,
        walk_length=0,
        prefer_latest_list=[False],
        mask_prob_max=0.0,
        embedder=config["embedder"],
        d_text=config["d_text"],
        shuffle_seed=0,
        context_seed=0,
        items_per_task=10_000_000,
        quiet=True,
        ignore_data_errors=False,
        mmap_populate=True,
        timeout_per_item=600.0,
        vector_db_path=None,
        db_cutoff=None,
        legacy_boolean=False,
    )


def featurize_task(*, db, table, pre_dir, ckpt, features_root, context_rows, context_seed, local_ctx, bfs_width, compile):
    from expts.repaper.baselines.rel2tabv2.pool_features import query_rows
    from rt.data import get_tasks, process_batch
    from rt.model import load_rt_model

    out = blob_dir(features_root, db, table)
    want = {"rtj_ckpt": ckpt, "context_rows": context_rows, "context_seed": context_seed,
            "local_ctx": local_ctx, "bfs_width": bfs_width}
    if (out / "meta.json").is_file():
        meta = json.loads((out / "meta.json").read_text())
        assert {k: meta[k] for k in want} == want, f"{out} was featurized for {meta}, not {want}"
        print(f"{out} exists; not refeaturizing", flush=True)
        return
    device = "cuda"
    t0 = time.perf_counter()
    ctx, rest = query_rows(pre_dir, db, table, context_rows, context_seed)
    nodes = np.concatenate([ctx, rest])
    net, config = load_rt_model(ckpt, device=device, compile=compile)
    net = net.to(torch.bfloat16).eval()
    (task,) = get_tasks(pre_dir, [(db, table)], ("test",))
    ds = make_dataset(task, pre_dir, config, local_ctx, bfs_width)
    rows = CELLS_PER_BATCH // local_ctx
    print(f"{db}/{table}: {len(ctx)} context + {len(rest)} other rows, local_ctx {local_ctx}, bfs_width {bfs_width}, "
          f"{rows} rows/batch, setup {time.perf_counter() - t0:.0f}s", flush=True)

    feats = np.empty((len(nodes), net.d_model), dtype=np.float32)
    n_real = np.empty(len(nodes), dtype=np.int32)
    n_nodes = np.empty(len(nodes), dtype=np.int32)
    max_depth = np.empty(len(nodes), dtype=np.int32)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(nodes), rows):
            chunk = nodes[start : start + rows]
            k = len(chunk)
            if k < rows:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], rows - k)])
            tup = ds.sampler.batch_for_nodes_py([int(v) for v in chunk], 0, local_ctx)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            batch = {key: v.to(device, non_blocking=True) for key, v in batch.items()}
            x = net(batch, return_embeddings=True)
            keys = batch["col_name_idxs"].masked_fill(batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max)
            si = keys.argsort(dim=-1, stable=True)
            is_t = batch["is_targets"].gather(1, si)
            assert (is_t.sum(1) == 1).all(), f"{db}/{table}: not one target per row"
            got = (batch["node_idxs"].gather(1, si) * is_t).sum(1)
            assert bool((got == torch.as_tensor(chunk, device=device, dtype=got.dtype)).all()), (
                f"{db}/{table}: sampler substituted a row near {start}"
            )
            feats[start : start + k] = x[is_t.bool()][:k].float().cpu().numpy()
            real = ~batch["is_padding"]
            ni = batch["node_idxs"].long().masked_fill(~real, -1).sort(dim=1).values
            distinct = ((ni[:, 1:] != ni[:, :-1]) & (ni[:, 1:] >= 0)).sum(1) + (ni[:, 0] >= 0).long()
            n_real[start : start + k] = real.sum(1)[:k].cpu().numpy()
            n_nodes[start : start + k] = distinct[:k].cpu().numpy()
            max_depth[start : start + k] = batch["bfs_depths"].masked_fill(~real, 0).amax(1)[:k].cpu().numpy()
            if (start // rows) % 100 == 0:
                done = start + k
                print(f"  {done}/{len(nodes)} rows, {done / (time.perf_counter() - t0):.0f} rows/s", flush=True)
    assert np.isfinite(feats).all(), f"{db}/{table}: non-finite features"
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(nodes, kind="stable")
    np.save(out / "node_idxs.npy", nodes[order].astype(np.int64))
    np.save(out / "target.npy", feats[order])
    np.savez(out / "ctx_stats.npz", node_idxs=nodes[order], n_real=n_real[order], n_nodes=n_nodes[order], max_depth=max_depth[order])
    q = lambda a: [float(v) for v in np.quantile(a, [0.05, 0.5, 0.95])]
    (out / "meta.json").write_text(json.dumps({
        **want, "compile": compile, "n_context": len(ctx), "n_rows": len(nodes), "n_features": int(net.d_model),
        "real_cells_p5_p50_p95": q(n_real), "real_cells_mean": float(n_real.mean()),
        "distinct_rows_p5_p50_p95": q(n_nodes), "distinct_rows_mean": float(n_nodes.mean()),
        "max_bfs_depth_p5_p50_p95": q(max_depth), "max_bfs_depth_mean": float(max_depth.mean()),
        "seconds": time.perf_counter() - t0,
    }, indent=1))
    print(f"{db}/{table}: wrote {len(nodes)} rows x {net.d_model} to {out} in {time.perf_counter() - t0:.0f}s; "
          f"real cells mean {n_real.mean():.0f}, distinct rows mean {n_nodes.mean():.1f}, max depth mean {max_depth.mean():.2f}",
          flush=True)
    del net
    torch.cuda.empty_cache()


class RowsFeaturizer:
    def __init__(self, features_root, db_tables):
        self._features = {}
        for db, table in sorted(set(db_tables)):
            d = blob_dir(features_root, db, table)
            meta = json.loads((d / "meta.json").read_text())
            idx = np.load(d / "node_idxs.npy")
            vec = torch.from_numpy(np.load(d / "target.npy"))
            self._features[db, table] = (idx, vec)
            print(f"RowsFeaturizer[local_ctx {meta['local_ctx']}, bfs_width {meta['bfs_width']}]: {db}/{table} "
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
