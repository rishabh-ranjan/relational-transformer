import json
from pathlib import Path


def main(
    *,
    features_root: str,
    tabpfn_dir: str,
    d_feat: int,
    min_rows: int,
    n_ctx_lo: int,
    n_ctx_hi: int,
    n_query: int,
    tasks_per_step: int,
    n_steps: int,
    sweep_tasks: list[tuple[str, str]],
    sweep_n_ctx: list[int],
    sweep_n_query: list[int],
    trace_n_ctx: int,
    seed: int,
) -> None:
    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from torch import nn

    from expts.repaper.adapter.train_adapter import Pool, draw, load_index

    # Every step of the v2 run clips: the pre-clip norm is 9.4-48.4 against a
    # grad_norm_max of 1.0. Either 20 is what this objective weighs, in which
    # case the threshold is the bug, or it is manufactured somewhere. dL/dW is
    # sum over rows of x_row (outer) dL/dx_row, so a hook on the adapter output
    # splits the total into the three things it can come from: how many rows,
    # how big the input rows are, and how big the per-row output gradient is.
    device = "cuda"
    root = Path(features_root).expanduser()
    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"

    def tabpfn(task_type):
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
            inference_precision=torch.float32,
            inference_config={"OUTLIER_REMOVAL_STD": None},
        )
        warm = torch.randn(32, d_feat, device=device)
        y = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
        if task_type == "reg":
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        for model in est.models_:
            for p in model.parameters():
                p.requires_grad_(False)
        return est

    ests = {"clf": tabpfn("clf"), "reg": tabpfn("reg")}
    adapter = nn.Linear(d_feat, d_feat, bias=False).to(device)
    with torch.no_grad():
        adapter.weight.copy_(torch.eye(d_feat))
    weight = adapter.weight

    def pct(t, qs=(0.5, 0.9, 0.99, 1.0)):
        v = t.double().flatten()
        return [float(torch.quantile(v, q)) for q in qs]

    def measure(task_type, emb, y, n_ctx):
        adapter.zero_grad(set_to_none=True)
        xin = emb.to(device).float()
        yy = y.to(device)
        est = ests[task_type]
        if task_type == "reg" and float(yy[:n_ctx].std()) < 1e-6:
            return None
        x = adapter(xin)
        if float(x[:n_ctx].std(dim=0).max()) == 0.0:
            return None
        seen = {}
        x.register_hook(lambda g: seen.__setitem__("dx", g.detach()))
        if task_type == "clf":
            lab = (yy > 0).long()
            est.fit_with_differentiable_input(x[:n_ctx], lab[:n_ctx])
            logits = est.forward(x[n_ctx:], use_inference_mode=True, return_logits=True)
            loss = nn.functional.cross_entropy(logits, lab[n_ctx:])
        else:
            est.fit_with_differentiable_input(x[:n_ctx], yy[:n_ctx])
            logits, _per, _b = est.forward(x[n_ctx:], use_inference_mode=True)
            z = (yy[n_ctx:] - est.y_train_mean_) / est.y_train_std_
            loss = est.znorm_space_bardist_(
                logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
            ).mean()
        if not torch.isfinite(loss):
            return None
        loss.backward()
        g = weight.grad.detach().clone()
        dx = seen["dx"]
        n_q = len(yy) - n_ctx
        gr, xr = dx.norm(dim=1).double(), xin.norm(dim=1).double()
        # dL/dW = sum_r x_r (outer) g_r. The sum of the per-row Frobenius norms
        # is the triangle bound; the root-sum-square is what it would be if the
        # rows' outer products were mutually orthogonal. Where the true norm
        # sits between them says whether the rows agree or cancel.
        gw_ctx = dx[:n_ctx].T @ xin[:n_ctx]
        gw_q = dx[n_ctx:].T @ xin[n_ctx:]
        sq = gr.square()
        top = torch.topk(sq, min(10, len(sq))).values
        return {
            "n_ctx": n_ctx,
            "n_query": n_q,
            "n_rows": len(yy),
            "loss": float(loss),
            "gw": float(g.norm()),
            "gw_ctx": float(gw_ctx.norm()),
            "gw_query": float(gw_q.norm()),
            "gw_split_err": float((gw_ctx + gw_q - g).abs().max()),
            "dx": float(dx.norm()),
            "dx_ctx": float(dx[:n_ctx].norm()),
            "dx_query": float(dx[n_ctx:].norm()),
            "dx_row_mean": float(gr.mean()),
            "dx_row_pct": pct(gr),
            "dx_ctx_row_mean": float(gr[:n_ctx].mean()),
            "dx_query_row_mean": float(gr[n_ctx:].mean()),
            "top1_share": float(top[0] / sq.sum()),
            "top10_share": float(top.sum() / sq.sum()),
            "x_row_mean": float(xr.mean()),
            "x_row_pct": pct(xr),
            # The z-norm divides column j by the context's std of column j, so
            # 1/std is a per-column gradient amplification straight into dL/dx.
            "x_dim_mean_absmax": float(xin[:n_ctx].mean(0).abs().max()),
            "x_dim_std_pct": pct(xin[:n_ctx].std(0), (0.0, 0.5, 1.0)),
            "inv_std_p50": float(1.0 / torch.quantile(xin[:n_ctx].std(0).double(), 0.5)),
            "bound_sum": float((gr * xr).sum()),
            "bound_rss": float((gr * xr).square().sum().sqrt()),
            "flat": g.flatten(),
        }

    def show(tag, r):
        print(
            f"{tag} n_ctx {r['n_ctx']:>5} n_q {r['n_query']:>4} loss {r['loss']:>10.4f} "
            f"|dW| {r['gw']:>8.3f} (ctx {r['gw_ctx']:>8.3f} q {r['gw_query']:>8.3f}) "
            f"|dx| {r['dx']:>9.4g} (ctx {r['dx_ctx']:>9.4g} q {r['dx_query']:>9.4g}) "
            f"|x|row {r['x_row_mean']:>7.3f} "
            f"rss {r['bound_rss']:>8.3f} sum {r['bound_sum']:>9.3f} "
            f"dW/(|x|*|dx|) {r['gw'] / (r['x_row_mean'] * r['dx']):>6.3f}",
            flush=True,
        )

    print("\n########## 1-2. real batches, per-task decomposition ##########", flush=True)
    entries = load_index(features_root, min_rows)
    pool = Pool(entries)
    rng = np.random.default_rng(seed)
    by_type = {"clf": [], "reg": []}
    per_step = []
    for step in range(n_steps):
        got = []
        for _ in range(tasks_per_step):
            e = entries[int(rng.choice(len(entries), p=pool.p))]
            emb, y, n_ctx = draw(rng, pool, e, n_ctx_lo, n_ctx_hi, n_query)
            r = measure(e["task_type"], emb, y, n_ctx)
            if r is None:
                print(f"step {step} {e['db']}/{e['task']}: skipped", flush=True)
                continue
            r["name"] = f"{e['db']}/{e['task']}"
            r["task_type"] = e["task_type"]
            got.append(r)
            by_type[e["task_type"]].append(r)
            show(f"step {step:>2} {r['task_type']} {r['name'][:46]:<46}", r)
        if not got:
            continue
        # The optimiser sees sum_i grad(loss_i)/tasks_per_step, and accumulation
        # is linear, so the step's gradient is exactly this sum -- no need to
        # rerun the four forwards to get it.
        total = torch.stack([r["flat"] for r in got]).sum(0) / tasks_per_step
        tri = sum(r["gw"] for r in got) / tasks_per_step
        share = [r["gw"] / tasks_per_step / float(total.norm()) for r in got]
        print(
            f"step {step:>2} TOTAL |sum/4| {float(total.norm()):>8.3f} "
            f"sum of |g_i|/4 {tri:>8.3f} (cancellation {float(total.norm()) / tri:.3f}) "
            f"per-task share {['%.2f' % s for s in share]} "
            f"max/min {max(share) / min(share):.2f}",
            flush=True,
        )
        cos = [
            float(
                torch.dot(a["flat"], b["flat"])
                / (a["flat"].norm() * b["flat"].norm())
            )
            for i, a in enumerate(got)
            for b in got[i + 1 :]
        ]
        print(f"step {step:>2}   pairwise cos {['%+.3f' % c for c in cos]}", flush=True)
        per_step.append(float(total.norm()))
        for r in got:
            r.pop("flat")

    print("\n---------- distribution over steps ----------", flush=True)
    a = np.array(per_step)
    print(
        f"step grad norm: n {len(a)} min {a.min():.3f} p50 {np.median(a):.3f} "
        f"p90 {np.quantile(a, 0.9):.3f} max {a.max():.3f} mean {a.mean():.3f}",
        flush=True,
    )
    for tt, rs in by_type.items():
        if not rs:
            continue
        g = np.array([r["gw"] for r in rs])
        n = np.array([r["n_rows"] for r in rs])
        dxn = np.array([r["dx"] for r in rs])
        ls = np.array([r["loss"] for r in rs])
        print(
            f"{tt}: n {len(rs)} loss p50 {np.median(ls):+.4f} "
            f"|dW| p50 {np.median(g):.3f} mean {g.mean():.3f} "
            f"min {g.min():.3f} max {g.max():.3f} "
            f"|dx| p50 {np.median(dxn):.4g} "
            f"|dW|/rows p50 {np.median(g / n):.5f} "
            f"|dW|/sqrt(rows) p50 {np.median(g / np.sqrt(n)):.4f}",
            flush=True,
        )
        print(
            f"{tt}: ctx share of |dW| p50 "
            f"{np.median([r['gw_ctx'] / r['gw'] for r in rs]):.3f} "
            f"per-row |dx| ctx p50 {np.median([r['dx_ctx_row_mean'] for r in rs]):.4g} "
            f"query p50 {np.median([r['dx_query_row_mean'] for r in rs]):.4g} "
            f"top1 row share p50 {np.median([r['top1_share'] for r in rs]):.4f} "
            f"top10 {np.median([r['top10_share'] for r in rs]):.4f}",
            flush=True,
        )

    print("\n########## 3-4. row-count sweep on fixed tasks ##########", flush=True)

    def load(db, task, n_take):
        m = json.loads((root / db / f"{task}_meta.json").read_text())
        n, d = m["shape"]
        arr = np.memmap(
            root / db / f"{task}_target.bin",
            dtype=ml_dtypes.bfloat16,
            mode="r",
            shape=(n, d),
        )
        lab = np.load(root / db / f"{task}_rows.npz")["labels"]
        idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n_take, n), replace=False))
        emb = torch.from_numpy(np.ascontiguousarray(arr[idx]).view(np.uint16)).view(
            torch.bfloat16
        )
        return emb, torch.from_numpy(lab[idx].astype("float32")), m["task_type"]

    for db, task in sweep_tasks:
        # One draw, reused: the query rows are byte-identical at every n_ctx, so
        # what moves between rows of the sweep is the context size and nothing
        # else.
        need = max(sweep_n_ctx) + max(sweep_n_query)
        emb, y, task_type = load(db, task, need)
        avail = len(y)
        print(f"\n=== {db}/{task} [{task_type}] {avail} rows drawn", flush=True)
        n_q0 = max(sweep_n_query)
        ctx_e, ctx_y = emb[: avail - n_q0], y[: avail - n_q0]
        q_e, q_y = emb[avail - n_q0 :], y[avail - n_q0 :]
        print("-- n_ctx sweep, n_query fixed --", flush=True)
        prev = None
        for n_c in sweep_n_ctx:
            if n_c > len(ctx_y):
                print(f"  n_ctx {n_c}: only {len(ctx_y)} context rows available", flush=True)
                continue
            r = measure(
                task_type,
                torch.cat([ctx_e[:n_c], q_e[:n_query]]),
                torch.cat([ctx_y[:n_c], q_y[:n_query]]),
                n_c,
            )
            if r is None:
                print(f"  n_ctx {n_c}: skipped", flush=True)
                continue
            r.pop("flat")
            show("  ", r)
            print(
                f"     x per-dim |mean| max {r['x_dim_mean_absmax']:.4g} "
                f"per-dim std min/p50/max {['%.4g' % v for v in r['x_dim_std_pct']]} "
                f"1/std(p50) {r['inv_std_p50']:.4g}",
                flush=True,
            )
            print(
                f"     row |dx| pct50/90/99/max {['%.3g' % v for v in r['dx_row_pct']]} "
                f"row |x| pct50/90/99/max {['%.3g' % v for v in r['x_row_pct']]} "
                f"top1 {r['top1_share']:.4f} top10 {r['top10_share']:.4f} "
                f"ctx/q per-row |dx| {r['dx_ctx_row_mean']:.4g}/{r['dx_query_row_mean']:.4g} "
                f"split_err {r['gw_split_err']:.3g}",
                flush=True,
            )
            if prev:
                print(
                    f"     vs previous: rows x{r['n_rows'] / prev['n_rows']:.2f} "
                    f"|dW| x{r['gw'] / prev['gw']:.2f} |dx| x{r['dx'] / prev['dx']:.2f}",
                    flush=True,
                )
            prev = r
        print("-- n_query sweep, n_ctx fixed at 512 --", flush=True)
        prev = None
        for n_qq in sweep_n_query:
            r = measure(
                task_type,
                torch.cat([ctx_e[:512], q_e[:n_qq]]),
                torch.cat([ctx_y[:512], q_y[:n_qq]]),
                512,
            )
            if r is None:
                print(f"  n_query {n_qq}: skipped", flush=True)
                continue
            r.pop("flat")
            show("  ", r)
            if prev:
                print(
                    f"     vs previous: n_q x{r['n_query'] / prev['n_query']:.2f} "
                    f"|dW| x{r['gw'] / prev['gw']:.2f} |dx| x{r['dx'] / prev['dx']:.2f}",
                    flush=True,
                )
            prev = r

    print("\n########## 5. layer-by-layer trace inside tabpfn ##########", flush=True)

    def tensors(o):
        if torch.is_tensor(o):
            yield o
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from tensors(v)
        elif isinstance(o, dict):
            for v in o.values():
                yield from tensors(v)

    def trace(db, task, emb, y, task_type, n_ctx):
        import tabpfn.preprocessing.steps.differentiable_z_norm_step as zmod

        est = ests[task_type]
        model = est.models_[0]
        rec, order_b, seen_b, handles = {}, [], set(), []

        def acc(r, key, t):
            t = t.detach().double()
            r[key + "_sq"] = r.get(key + "_sq", 0.0) + float(t.square().sum())
            r[key + "_max"] = max(r.get(key + "_max", 0.0), float(t.abs().max()))

        def make(name):
            def fwd(m, inp, out):
                r = rec.setdefault(
                    name, {"type": type(m).__name__, "calls": 0, "shape": None}
                )
                r["calls"] += 1
                for t in tensors(out):
                    acc(r, "act", t)
                    r["shape"] = tuple(t.shape)
                    if t.requires_grad:

                        def go(g, r=r, name=name):
                            acc(r, "go", g)
                            if name not in seen_b:
                                seen_b.add(name)
                                order_b.append(name)
                            return None

                        t.register_hook(go)
                # A tensor consumed by two modules (a residual stream) gets the
                # sum of both paths here, so gi is the gradient of that tensor,
                # not of this module's share of it.
                for t in tensors(inp):
                    if t.requires_grad:

                        def gi(g, r=r):
                            acc(r, "gi", g)
                            return None

                        t.register_hook(gi)

            return fwd

        named = list(model.named_modules())
        for name, mod in named:
            handles.append(mod.register_forward_hook(make(name or "<model>")))

        # The z-norm is a PreprocessingStep, not an nn.Module, so it has no hook
        # to register; patching its transform is the only way to see the tensor
        # crossing from preprocessing into the transformer.
        zstats, orig = [], zmod.DifferentiableZNormStep._transform

        def patched(self, X, *a, **kw):
            out = orig(self, X, *a, **kw)
            xo = out[0]
            s = {
                "is_test": kw.get("is_test", a[0] if a else None),
                "shape": tuple(X.shape),
                "in_norm": float(X.detach().double().norm()),
                "in_absmax": float(X.detach().abs().max()),
                "in_dim_mean_absmax": float(X.detach().mean(0).abs().max()),
                "in_dim_std": pct(X.detach().std(0), (0.0, 0.5, 1.0)),
                "out_norm": float(xo.detach().double().norm()),
                "out_absmax": float(xo.detach().abs().max()),
                "std_used": pct(self.stds.detach().flatten(), (0.0, 0.5, 1.0)),
                "mean_used_absmax": float(self.means.detach().abs().max()),
            }
            zstats.append(s)
            if xo.requires_grad:

                def h(g, s=s):
                    s["g_out"] = float(g.detach().double().norm())
                    return None

                xo.register_hook(h)
            if X.requires_grad:

                def h2(g, s=s):
                    s["g_in"] = float(g.detach().double().norm())
                    return None

                X.register_hook(h2)
            return out

        zmod.DifferentiableZNormStep._transform = patched
        try:
            r = measure(task_type, emb, y, n_ctx)
        finally:
            zmod.DifferentiableZNormStep._transform = orig
            for h in handles:
                h.remove()
        if r is None:
            print(f"{db}/{task}: skipped", flush=True)
            return
        r.pop("flat")
        print(f"\n=== {db}/{task} [{task_type}] {len(named)} modules", flush=True)
        show("  ", r)
        print(
            f"  x per-dim |mean| max {r['x_dim_mean_absmax']:.4g} "
            f"per-dim std min/p50/max {['%.4g' % v for v in r['x_dim_std_pct']]}",
            flush=True,
        )
        print("\n-- preprocessing boundary: DifferentiableZNormStep --", flush=True)
        for s in zstats:
            print(
                f"  is_test {s['is_test']} shape {s['shape']}\n"
                f"    in  |X| {s['in_norm']:.4g} absmax {s['in_absmax']:.4g} "
                f"per-dim |mean| max {s['in_dim_mean_absmax']:.4g} "
                f"per-dim std min/p50/max {['%.4g' % v for v in s['in_dim_std']]}\n"
                f"    out |X| {s['out_norm']:.4g} absmax {s['out_absmax']:.4g} "
                f"(fitted std min/p50/max {['%.4g' % v for v in s['std_used']]} "
                f"|mean| max {s['mean_used_absmax']:.4g})\n"
                f"    grad out {s.get('g_out')} in {s.get('g_in')} "
                f"amplification "
                f"{(s['g_in'] / s['g_out']) if s.get('g_out') else None}",
                flush=True,
            )

        rows = []
        for name in order_b:
            r2 = rec[name]
            go = r2.get("go_sq", 0.0) ** 0.5
            gi = r2.get("gi_sq", 0.0) ** 0.5
            rows.append(
                {
                    "name": name,
                    "type": r2["type"],
                    "calls": r2["calls"],
                    "shape": r2["shape"],
                    "act": r2.get("act_sq", 0.0) ** 0.5,
                    "act_max": r2.get("act_max", 0.0),
                    "go": go,
                    "gi": gi,
                    "ratio": (gi / go) if go > 0 else float("nan"),
                }
            )
        silent = [n for n, _ in named if (n or "<model>") not in seen_b]
        print(
            f"\n-- {len(rows)} modules fired a backward hook, "
            f"{len(silent)} silent (no grad through their output) --",
            flush=True,
        )
        if silent:
            print(f"  silent: {silent[:20]}{' ...' if len(silent) > 20 else ''}", flush=True)

        def table(rs, title):
            print(f"\n-- {title} --", flush=True)
            print(
                f"  {'module':<58} {'type':<28} {'act':>10} {'actmax':>9} "
                f"{'|go|':>10} {'|gi|':>10} {'gi/go':>8} calls",
                flush=True,
            )
            for r2 in rs:
                print(
                    f"  {r2['name'][-58:]:<58} {r2['type'][:28]:<28} "
                    f"{r2['act']:>10.4g} {r2['act_max']:>9.4g} "
                    f"{r2['go']:>10.4g} {r2['gi']:>10.4g} "
                    f"{r2['ratio']:>8.3f} {r2['calls']}",
                    flush=True,
                )

        table(
            [r2 for r2 in rows if r2["name"].count(".") <= 2],
            "backward order, top of the tree (depth <= 2)",
        )
        table(
            sorted(rows, key=lambda r2: -r2["ratio"])[:30],
            "largest per-layer amplification gi/go, any depth",
        )
        table(
            sorted(rows, key=lambda r2: -r2["act_max"])[:20],
            "largest activation absmax, any depth",
        )
        table(
            sorted(rows, key=lambda r2: -r2["go"])[:20],
            "largest incoming gradient |go|, any depth",
        )

        # register_full_backward_hook is what one reaches for first; it wraps the
        # module's inputs and outputs in autograd nodes, which several of these
        # modules -- tuple returns with a None in them -- do not survive. Run it
        # separately so a failure costs the second pass and not the trace.
        fired, hs = {}, []

        def bwd(name):
            def h(m, gin, gout):
                a = sum(
                    float(t.detach().double().square().sum())
                    for t in tensors(gin)
                    if t is not None
                )
                b = sum(
                    float(t.detach().double().square().sum())
                    for t in tensors(gout)
                    if t is not None
                )
                fired[name] = (a**0.5, b**0.5)

            return h

        try:
            for name, mod in named:
                hs.append(mod.register_full_backward_hook(bwd(name or "<model>")))
            r3 = measure(task_type, emb, y, n_ctx)
            print(
                f"\n-- register_full_backward_hook: fired for {len(fired)}/"
                f"{len(named)} modules, |dW| {r3 and r3['gw']} "
                f"(tensor-hook pass gave {r['gw']:.4g}) --",
                flush=True,
            )
            by_name = {x["name"]: x for x in rows}
            for n in list(fired)[:15]:
                if n not in by_name:
                    continue
                a, b = fired[n]
                print(
                    f"  {n[-50:]:<50} fbh gi {a:>10.4g} go {b:>10.4g} | "
                    f"tensor gi {by_name[n]['gi']:>10.4g} "
                    f"go {by_name[n]['go']:>10.4g}",
                    flush=True,
                )
        except Exception as ex:
            print(f"\n-- register_full_backward_hook FAILED: {type(ex).__name__}: {ex}", flush=True)
        finally:
            for h in hs:
                h.remove()

    for db, task in sweep_tasks:
        emb, y, task_type = load(db, task, trace_n_ctx + n_query)
        trace(db, task, emb, y, task_type, len(y) - n_query)
