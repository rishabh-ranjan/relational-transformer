import json
import shutil
import zlib
from pathlib import Path


def listed_tasks(pre_dir: str, db: str, db_task_list: str) -> list[tuple[str, object]]:
    from rt.data import get_tasks, resolve_db_task_list

    meta = json.loads((Path(pre_dir).expanduser() / db / "meta.json").read_text())
    known = {t["name"] for t in meta.get("tasks", [])}
    names = sorted(
        n for d, n in resolve_db_task_list(db_task_list) if d == db and n in known
    )
    # One call per name, not one call for all of them: get_tasks keys an
    # autocomplete task by its entity table, so two of them on the same table
    # come back indistinguishable and the task name -- which is what a dump
    # file is named after -- is lost.
    out = []
    for name in names:
        got = get_tasks(pre_dir, [(db, name)], ("train",))
        if got:
            out.append((name, got[0]))
    return out


def pool_rows(np, pre_dir, db, name, table, shuffle_seed, max_rows, **_):
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    lo, n = table_offset_and_len(pre_dir, db, table)
    if n <= max_rows:
        return np.arange(lo, lo + n), n
    # A uniform draw, not a prefix: node idx order is time order, so the first
    # max_rows rows of a big task table are its earliest horizons rather than a
    # sample of it.
    # crc32, not hash(): python salts str hashes per process, so hash() here
    # would draw different rows on a rerun.
    key = f"{shuffle_seed}/{db}/{name}".encode()
    rng = np.random.default_rng(zlib.crc32(key))
    return np.sort(rng.choice(n, size=max_rows, replace=False) + lo), n


def featurize_dbs(
    *,
    dbs: list[str],
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt: str,
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    batch_size: int,
    min_rows: int,
    max_rows: int,
    expected_gib: float,
) -> None:
    import time

    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch

    from rt.data import RustlerDataset, process_batch
    from rt.model import load_rt_model

    device = "cuda"
    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()
    d_model = net.d_model

    # return_embeddings=True is norm_out over the last block: the post-norm_out
    # unit-12 target cell, and exactly how features_rt-j was built. So the Join
    # side of the adapter's input and the RelBench side are the same
    # construction, not merely the same layer, and RelBench needs no dump of its
    # own -- rt_features already holds every row of all 21 tasks.

    total_written = 0
    t_start = time.time()
    for db in dbs:
        named = listed_tasks(pre_dir, db, db_task_list)
        if not named:
            print(f"[{db}] no listed task in this build, skipping", flush=True)
            continue
        out_dir = Path(features_root).expanduser() / db
        out_dir.mkdir(parents=True, exist_ok=True)

        # One sampler for the whole db: batch_for_nodes_py takes a dataset_idx
        # into the task list, so the mmap populate is paid once per database
        # rather than once per task, and a db carries up to 276 of them.
        ds = RustlerDataset(
            tasks=[t for _n, t in named],
            pre_dir=pre_dir,
            global_rank=0,
            local_rank=0,
            world_size=1,
            local_ctx_size_list=[local_ctx_size],
            bfs_width_list=[bfs_width],
            num_walks=0,
            walk_length=0,
            prefer_latest_list=[False],
            mask_prob_max=0.0,
            embedder=config["embedder"],
            d_text=config["d_text"],
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            items_per_task=10_000_000,
            quiet=True,
            ignore_data_errors=False,
            mmap_populate=True,
            # Not 3600: check_deadline is the only bound on a pathological
            # BFS, and at an hour a single row can hold a batch -- and so a
            # gpu -- indefinitely, which is what hung two shards on
            # join-motogp/sessions-number and join-europeana/items-year. A row
            # that cannot be built in a minute is not worth having, and on
            # timeout it takes the same path as any other failure: panic,
            # substitute, and the node check below drops it.
            timeout_per_item=60.0,
            vector_db_path=None,
            db_cutoff=None,
            legacy_boolean=False,
        )

        db_written = 0
        n_done = n_skipped = db_sub = 0
        for dataset_idx, (name, task) in enumerate(named):
            bin_path = out_dir / f"{name}_target.bin"
            meta_path = out_dir / f"{name}_meta.json"
            if bin_path.exists() and meta_path.exists():
                n_done += 1
                continue

            rows, total_nodes = pool_rows(
                np,
                pre_dir=pre_dir,
                db=db,
                name=name,
                table=task.table_name,
                shuffle_seed=shuffle_seed,
                max_rows=max_rows,
            )
            if len(rows) < min_rows:
                n_skipped += 1
                continue
            n_rows = len(rows)

            need = n_rows * d_model * 2
            free = shutil.disk_usage(out_dir).free
            assert need * 1.1 < free, (
                f"[{db}] {name} needs {need / 2**20:.0f} MiB and only "
                f"{free / 2**30:.1f} GiB is free"
            )

            # rustler's seq() does not fail a row it cannot build: it swaps in
            # a *different* random item from the sampler's pool and carries on
            # (fly.rs:955). For training that is invisible and harmless -- items
            # are a random draw anyway -- but batch_for_nodes_py asks for
            # specific rows, so the swap silently files one row's embedding
            # under another row's node idx, and the substitute can belong to a
            # different task in the same db. Two of the three triggers are
            # silent: a target column the node does not carry, and a null target
            # value, both of which return BuildError into an empty match arm.
            # The third, >5 f2p neighbours, panics and prints. So the only
            # reliable check is to ask what node each row actually came from.
            kept_nodes, kept_labels = [], []
            n_sub = 0
            tmp = bin_path.with_suffix(".bin.partial")
            with open(tmp, "wb") as fh, torch.inference_mode():
                for start in range(0, n_rows, batch_size):
                    node_idxs = [int(v) for v in rows[start : start + batch_size]]
                    B = len(node_idxs)
                    tup = ds.sampler.batch_for_nodes_py(
                        node_idxs, dataset_idx, local_ctx_size
                    )
                    batch = process_batch(tup, ds.d_text)
                    batch.pop("batch_mask", None)
                    batch = {
                        k: v.to(device, non_blocking=True) for k, v in batch.items()
                    }
                    x = net(batch, return_embeddings=True)

                    # forward() sorts the sequence before the blocks run, so
                    # its output is already in the canonical frame; only
                    # is_targets, which comes off the unsorted batch, has to be
                    # permuted to meet it.
                    keys = batch["col_name_idxs"].masked_fill(
                        batch["is_padding"],
                        torch.iinfo(batch["col_name_idxs"].dtype).max,
                    )
                    si = keys.argsort(dim=-1, stable=True)
                    is_targets = batch["is_targets"].gather(1, si)
                    per_row = is_targets.sum(dim=1)
                    assert (per_row == 1).all(), (
                        f"{db}/{name}: expected one masked target cell per row, "
                        f"got counts {per_row.unique().tolist()}"
                    )
                    vals = (
                        batch["number_values"]
                        .gather(1, si.unsqueeze(-1).expand_as(batch["number_values"]))
                        .squeeze(-1)
                    )
                    labels = (
                        (vals * is_targets.to(vals.dtype))
                        .sum(dim=1)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    emb = x[is_targets].bfloat16()
                    assert emb.shape == (B, d_model), emb.shape

                    # The substitution check. The node the target cell actually
                    # belongs to must be the node we asked for; anything else is
                    # a row seq() swapped in, and keeping it would file one
                    # row's embedding under another row's idx.
                    got = (
                        (batch["node_idxs"].gather(1, si) * is_targets)
                        .sum(dim=1)
                        .cpu()
                        .numpy()
                    )
                    want = np.asarray(node_idxs, dtype=got.dtype)
                    ok = got == want
                    n_sub += int((~ok).sum())
                    fh.write(
                        emb[torch.from_numpy(ok).to(device)]
                        .view(torch.uint16)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
                    kept_nodes.append(want[ok])
                    kept_labels.append(labels[ok])

            kept = np.concatenate(kept_nodes) if kept_nodes else np.empty(0, np.int64)
            labels = (
                np.concatenate(kept_labels)
                if kept_labels
                else np.empty(0, np.float32)
            )
            if len(kept) < min_rows:
                tmp.unlink()
                print(
                    f"[{db}] {name}: only {len(kept)} of {n_rows} rows are "
                    f"buildable, under {min_rows}; skipping",
                    flush=True,
                )
                n_skipped += 1
                continue
            n_rows = len(kept)
            tmp.replace(bin_path)
            np.savez(
                out_dir / f"{name}_rows.npz",
                node_idxs=kept.astype(np.int64),
                labels=labels,
            )
            written = bin_path.stat().st_size
            db_written += written
            total_written += written
            finite = labels[np.isfinite(labels)]
            meta_path.write_text(
                json.dumps(
                    {
                        "dtype": "bfloat16",
                        "shape": [n_rows, d_model],
                        "task": name,
                        "table": task.table_name,
                        "task_type": task.task_type,
                        "target_column": task.target_column,
                        # The mixture weight: pretraining draws a row uniformly
                        # from a pool holding min(rows, 100_000) of every task,
                        # so P(task) is set by this, not by what we dumped.
                        "total_nodes": total_nodes,
                        "sampled_rows": n_rows,
                        # Rows seq() could not build and silently swapped out,
                        # dropped here rather than written under the wrong node.
                        "n_substituted": n_sub,
                        "label_mean": float(finite.mean()) if finite.size else None,
                        "label_std": float(finite.std()) if finite.size else None,
                        "label_frac_positive": float((finite > 0).mean())
                        if finite.size
                        else None,
                        "n_distinct_labels": int(np.unique(finite).size),
                        "n_label_nonfinite": int(labels.size - finite.size),
                        "local_ctx_size": local_ctx_size,
                        "bfs_width": bfs_width,
                        "context_seed": context_seed,
                        "shuffle_seed": shuffle_seed,
                        "normalisation": "norm_out over the last block, as features_rt-j",
                    },
                    indent=2,
                )
            )
            db_sub += n_sub
            n_done += 1

        gib = total_written / 2**30
        print(
            f"[{db}] {n_done} tasks, {n_skipped} skipped, {db_sub} substituted "
            f"rows dropped, {db_written / 2**20:.0f} MiB "
            f"| cumulative {gib:.2f} of an expected "
            f"{expected_gib:.2f} GiB, {(time.time() - t_start) / 60:.1f} min",
            flush=True,
        )
        assert gib <= expected_gib * 1.5 + 0.5, (
            f"wrote {gib:.2f} GiB against an expected {expected_gib:.2f} GiB; "
            f"stopping before the remaining dbs run the disk down"
        )

    print(
        f"done: {total_written / 2**30:.2f} GiB over {len(dbs)} dbs in "
        f"{(time.time() - t_start) / 60:.1f} min",
        flush=True,
    )
