import json
import math
import time
import zlib
from pathlib import Path


def load_index(features_root: str, min_rows: int, holdout_mod: int):
    import numpy as np

    root = Path(features_root).expanduser()
    train, held = [], []
    skipped = {"rows": 0, "degenerate": 0}
    for mp in sorted(root.glob("*/*_meta.json")):
        m = json.loads(mp.read_text())
        n, d = m["shape"]
        if n < min_rows:
            skipped["rows"] += 1
            continue
        # A task whose label never varies teaches nothing, and TabPFN's
        # regressor refuses a constant target outright.
        if (m["n_distinct_labels"] or 0) < 2:
            skipped["degenerate"] += 1
            continue
        db = mp.parent.name
        entry = {
            "db": db,
            "task": m["task"],
            "task_type": m["task_type"],
            "n": n,
            "d": d,
            "bin": str(mp.with_name(mp.name.replace("_meta.json", "_target.bin"))),
            "rows": str(mp.with_name(mp.name.replace("_meta.json", "_rows.npz"))),
            # P(task) as pretraining draws it: a pool holding min(rows, 100_000)
            # of every task, sampled uniformly. What we dumped is capped far
            # lower, so the weight has to come off the true row count.
            "weight": float(min(m["total_nodes"], 100_000)),
        }
        # Held out by database, not by row: an unseen schema is the thing the
        # adapter has to generalise to, and held-out rows of a task whose other
        # rows are the context is just the training objective measured again.
        # crc32, not hash(): python salts str hashes per process, so hash()
        # would reshuffle the holdout on every run and on every resume.
        (held if zlib.crc32(db.encode()) % holdout_mod == 0 else train).append(entry)
    print(
        f"index: {len(train)} train tasks, {len(held)} held-out tasks "
        f"({len({e['db'] for e in train})}/{len({e['db'] for e in held})} dbs), "
        f"skipped {skipped}",
        flush=True,
    )
    for split, es in (("train", train), ("held", held)):
        by = {t: sum(1 for e in es if e["task_type"] == t) for t in ("clf", "reg")}
        print(f"  {split}: {by}", flush=True)
    return train, held


class Pool:
    def __init__(self, entries):
        import numpy as np

        self.entries = entries
        w = np.array([e["weight"] for e in entries], dtype=np.float64)
        self.p = w / w.sum()
        self._cache = {}

    def read(self, e, idx):
        import ml_dtypes
        import numpy as np
        import torch

        key = e["bin"]
        if key not in self._cache:
            x = np.memmap(
                e["bin"], dtype=ml_dtypes.bfloat16, mode="r", shape=(e["n"], e["d"])
            )
            y = np.load(e["rows"])["labels"]
            self._cache[key] = (x, y)
        x, y = self._cache[key]
        emb = torch.from_numpy(np.ascontiguousarray(x[idx]).view(np.uint16)).view(
            torch.bfloat16
        )
        return emb, torch.from_numpy(y[idx].astype("float32"))


def draw(rng, pool, e, n_ctx_lo, n_ctx_hi, n_query):
    import numpy as np

    hi = min(n_ctx_hi, e["n"] - n_query)
    lo = min(n_ctx_lo, hi)
    n_ctx = int(math.exp(rng.uniform(math.log(lo), math.log(hi + 1))))
    n_ctx = max(lo, min(n_ctx, hi))
    idx = rng.choice(e["n"], size=n_ctx + n_query, replace=False)
    emb, y = pool.read(e, np.sort(idx))
    return emb, y, n_ctx


def main(
    *,
    features_root: str,
    tabpfn_dir: str,
    out_dir: str,
    d_feat: int,
    min_rows: int,
    holdout_mod: int,
    n_ctx_lo: int,
    n_ctx_hi: int,
    n_query: int,
    tasks_per_step: int,
    total_steps: int,
    lr: float,
    wd: float,
    warmup_steps: int,
    grad_norm_max: float,
    eval_every: int,
    eval_tasks: int,
    eval_n_ctx: int,
    save_every: int,
    seed: int,
) -> None:
    import numpy as np
    import torch
    from torch import nn
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    device = "cuda"
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"
    assert model_path.exists(), f"TabPFN checkpoint {model_path} not found"

    assert min_rows > n_query + 64, (
        f"min_rows {min_rows} leaves no context after {n_query} queries"
    )
    train_entries, held_entries = load_index(features_root, min_rows, holdout_mod)
    train_pool, held_pool = Pool(train_entries), Pool(held_entries)

    # Identity, not Xavier: step 0 is then exactly the un-adapted rt-j
    # featurizer, so every later step is attributable to training.
    adapter = nn.Linear(d_feat, d_feat, bias=True).to(device)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(d_feat))
        adapter.bias.zero_()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)

    def tabpfn(task_type):
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
        )
        warm = torch.randn(32, d_feat, device=device)
        y = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
        if task_type == "reg":
            # The standard fit moves the bar distribution to the device and the
            # differentiable one does not, so its borders sit on the cpu and the
            # loss dies in searchsorted.
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        for model in est.models_:
            for p in model.parameters():
                p.requires_grad_(False)
        return est

    ests = {"clf": tabpfn("clf"), "reg": tabpfn("reg")}
    print("tabpfn frozen; only the adapter trains", flush=True)

    def loss_and_pred(e, emb, y, n_ctx):
        x = adapter(emb.to(device).float())
        yy = y.to(device)
        est = ests[e["task_type"]]
        if e["task_type"] == "clf":
            lab = (yy > 0).long()
            est.fit_with_differentiable_input(x[:n_ctx], lab[:n_ctx])
            logits = est.forward(x[n_ctx:], use_inference_mode=True, return_logits=True)
            loss = nn.functional.cross_entropy(logits, lab[n_ctx:])
            return loss, torch.softmax(logits, -1)[:, 1], lab[n_ctx:]
        est.fit_with_differentiable_input(x[:n_ctx], yy[:n_ctx])
        logits, _per, _b = est.forward(x[n_ctx:], use_inference_mode=True)
        z = (yy[n_ctx:] - est.y_train_mean_) / est.y_train_std_
        loss = est.znorm_space_bardist_(
            logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
        ).mean()
        mean = est.znorm_space_bardist_.mean(logits.transpose(0, 1))
        return loss, mean * est.y_train_std_ + est.y_train_mean_, yy[n_ctx:]

    @torch.no_grad()
    def evaluate(pool, n_tasks, rng):
        from rt.eval.metrics import metric_for

        # clf is auroc. reg is MAE over the trivial constant predictor's MAE:
        # raw MAE is not comparable across tasks whose targets are on different
        # scales, and the mean of it over a held-out set would be whichever
        # task happens to have the widest target.
        scores = {"clf": [], "reg": []}
        order = rng.permutation(len(pool.entries))[:n_tasks]
        for i in order:
            e = pool.entries[int(i)]
            n_ctx = min(eval_n_ctx, e["n"] - n_query)
            if n_ctx < 64:
                continue
            idx = np.sort(
                np.random.default_rng(int(i)).choice(
                    e["n"], size=n_ctx + n_query, replace=False
                )
            )
            emb, y = pool.read(e, idx)
            _loss, pred, truth = loss_and_pred(e, emb, y, n_ctx)
            t = truth.float().cpu().numpy()
            p = pred.float().cpu().numpy()
            try:
                _n, v = metric_for(e["task_type"], t, p)
            except ValueError:
                continue
            if e["task_type"] == "reg":
                ctx_median = float(np.median(y[:n_ctx].numpy()))
                denom = float(np.abs(t - ctx_median).mean())
                if denom <= 0:
                    continue
                v = v / denom
            scores[e["task_type"]].append(float(v))
        return {k: (float(np.mean(v)) if v else None, len(v)) for k, v in scores.items()}

    rng = np.random.default_rng(seed)
    t0 = time.time()
    running = {"clf": [], "reg": []}
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup_steps))
        opt.zero_grad(set_to_none=True)
        # One task is one very correlated gradient, so a step is several of
        # them accumulated rather than a bigger draw from one task.
        for _ in range(tasks_per_step):
            e = train_entries[int(rng.choice(len(train_entries), p=train_pool.p))]
            emb, y, n_ctx = draw(rng, train_pool, e, n_ctx_lo, n_ctx_hi, n_query)
            loss, _pred, _truth = loss_and_pred(e, emb, y, n_ctx)
            (loss / tasks_per_step).backward()
            running[e["task_type"]].append(float(loss))
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_norm_max)
        opt.step()

        if step % 20 == 0:
            msg = " ".join(
                f"{k} {np.mean(v):.4f}" for k, v in running.items() if v
            )
            dw = float((adapter.weight - torch.eye(d_feat, device=device)).norm())
            print(
                f"step {step:>6} {msg} |W-I| {dw:.4f} "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
            running = {"clf": [], "reg": []}

        if eval_every and step % eval_every == 0:
            held = evaluate(held_pool, eval_tasks, np.random.default_rng(0))
            print(
                f"step {step:>6} held-out dbs: "
                f"auroc {held['clf'][0]} (n={held['clf'][1]}) "
                f"nmae {held['reg'][0]} (n={held['reg'][1]})",
                flush=True,
            )

        if save_every and step % save_every == 0:
            torch.save(
                {"step": step, "state_dict": adapter.state_dict()},
                out / f"adapter_step{step}.pt",
            )

    torch.save(
        {"step": total_steps, "state_dict": adapter.state_dict()},
        out / "adapter_final.pt",
    )
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
