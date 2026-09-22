import json
import os
import time
from functools import partial
from pathlib import Path


def sort_frame(batch):
    import torch

    from rt.model.net import SEM_TYPE_NAMES

    col = batch["col_name_idxs"]
    pad = batch["is_padding"]
    si = col.masked_fill(pad, torch.iinfo(col.dtype).max).argsort(dim=-1, stable=True)
    s3 = si.unsqueeze(-1)
    out = {
        "node_idxs": batch["node_idxs"].gather(1, si),
        "f2p_nbr_idxs": batch["f2p_nbr_idxs"].gather(1, s3.expand_as(batch["f2p_nbr_idxs"])),
        "col_name_idxs": col.gather(1, si),
        "table_name_idxs": batch["table_name_idxs"].gather(1, si),
        "is_padding": pad.gather(1, si),
        "col_name_values": batch["col_name_values"].gather(1, s3.expand_as(batch["col_name_values"])),
        "sem_types": batch["sem_types"].gather(1, si),
        "is_targets": batch["is_targets"].gather(1, si),
    }
    for t in SEM_TYPE_NAMES:
        k = t + "_values"
        out[k] = batch[k].gather(1, s3.expand_as(batch[k]))
    return out


def dense_masks(s):
    node, f2p, col, tab, pad_ = (
        s["node_idxs"], s["f2p_nbr_idxs"], s["col_name_idxs"], s["table_name_idxs"], s["is_padding"]
    )
    pad = (~pad_[:, :, None]) & (~pad_[:, None, :])
    same_node = node[:, :, None] == node[:, None, :]
    kv_in_f2p = (node[:, None, :, None] == f2p[:, :, None, :]).any(-1)
    q_in_f2p = (node[:, :, None, None] == f2p[:, None, :, :]).any(-1)
    same_ct = (col[:, :, None] == col[:, None, :]) & (tab[:, :, None] == tab[:, None, :])
    masks = {
        "feat": ((same_node | kv_in_f2p) & pad).contiguous(),
        "nbr": (q_in_f2p & pad).contiguous(),
        "col": (same_ct & pad).contiguous(),
    }
    kv = {k: m.sum(dim=-1, keepdim=True).bfloat16() for k, m in masks.items()}
    return masks, kv


def block_masks_from_dense(masks, block_size):
    from rt.model.net import _make_block_mask
    from torch.nn.attention.flex_attention import create_block_mask

    out = {}
    for k, m in masks.items():
        B, S, _ = m.shape
        if block_size == 128:
            out[k] = _make_block_mask(m, B, S, S, m.device)
        else:
            def mod(b, h, q, kv, m=m):
                return m[b, q, kv]

            out[k] = create_block_mask(mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=m.device,
                                       BLOCK_SIZE=block_size, _compile=True)
    return out


def block_masks_from_mods(s, block_size):
    from torch.nn.attention.flex_attention import create_block_mask

    node = s["node_idxs"].contiguous()
    f2p = s["f2p_nbr_idxs"].contiguous()
    col = s["col_name_idxs"].contiguous()
    tab = s["table_name_idxs"].contiguous()
    pad = s["is_padding"].contiguous()

    def feat(b, h, q, kv):
        np_ = (~pad[b, q]) & (~pad[b, kv])
        return ((node[b, q] == node[b, kv]) | (f2p[b, q] == node[b, kv]).any(dim=-1)) & np_

    def nbr(b, h, q, kv):
        np_ = (~pad[b, q]) & (~pad[b, kv])
        return (f2p[b, kv] == node[b, q]).any(dim=-1) & np_

    def colm(b, h, q, kv):
        np_ = (~pad[b, q]) & (~pad[b, kv])
        return (col[b, q] == col[b, kv]) & (tab[b, q] == tab[b, kv]) & np_

    B, S = node.shape
    mk = partial(create_block_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, device=node.device,
                 BLOCK_SIZE=block_size, _compile=True)
    return {"feat": mk(feat), "nbr": mk(nbr), "col": mk(colm)}


def kv_sizes_fast(s):
    from rt.model.net import _kv_sizes

    return _kv_sizes(s["node_idxs"].contiguous(), s["f2p_nbr_idxs"].contiguous(),
                     s["col_name_idxs"].contiguous(), s["table_name_idxs"].contiguous(),
                     s["is_padding"].contiguous())


def encode(net, s):
    from rt.model.net import SEM_TYPE_NAMES

    pad = s["is_padding"]
    x = net.norm_dict["col_name"](net.enc_dict["col_name"](s["col_name_values"])) * (~pad)[..., None]
    for i, t in enumerate(SEM_TYPE_NAMES):
        st = s["sem_types"] == i
        x = x + net.norm_dict[t](net.enc_dict[t](s[t + "_values"])) * (st & ~s["is_targets"] & ~pad)[..., None]
        x = x + net.mask_embs[t] * (st & s["is_targets"] & ~pad)[..., None]
    return x


def run_blocks(net, x, bms, kv):
    for blk in net.blocks:
        x = blk(x, bms, kv)
    return net.norm_out(x)


def run_blocks_sdpa(net, x, kv):
    import torch.nn.functional as F
    from einops import rearrange

    for blk in net.blocks:
        for a in blk.attn_types:
            m = blk.attns[a]
            h = blk.norms[a](x)
            q = rearrange(m.wq(h), "b s (h d) -> b h s d", h=m.num_heads)
            k = rearrange(m.wk(h), "b s (h d) -> b h s d", h=m.num_heads)
            v = rearrange(m.wv(h), "b s (h d) -> b h s d", h=m.num_heads)
            q = m.q_norm(q) * m.scale
            k = m.k_norm(k)
            o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / m.head_dim)
            o = rearrange(o, "b h s d -> b s (h d)")
            x = x + m.wo(2 * F.sigmoid(m.wg(h)) * o)
        x = x + blk.ffn(blk.norms["ffn"](x))
    return net.norm_out(x)


def block_stats(bms):
    out = {}
    for k, bm in bms.items():
        tot = bm.kv_num_blocks.shape[-1] * bm.kv_indices.shape[-1]
        part = bm.kv_num_blocks.float().sum(-1).mean().item()
        full = bm.full_kv_num_blocks.float().sum(-1).mean().item() if bm.full_kv_num_blocks is not None else 0.0
        out[k] = {"blocks_per_seq": tot, "partial": part, "full": full, "empty": tot - part - full}
    return out


def main(
    *,
    features_root: str,
    pre_dir: str,
    ckpt: str,
    out_dir: str,
    n_tasks: int,
    rows_per_task: int,
    batch_size: int,
    reps: int,
    train_batch_size: int,
    seed: int,
) -> None:
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile

    from expts.repaper.adapter.probe_pool_throughput import make_dataset, resolve_tasks, sample
    from expts.repaper.adapter.train_adapter import load_index
    from rt.model import load_rt_model

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "batch_size": batch_size}

    def dump():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    def log(msg):
        print(msg, flush=True)

    entries = list(load_index(features_root, rows_per_task))
    w = np.array([e["weight"] for e in entries])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(entries), size=n_tasks, replace=False, p=w / w.sum())
    chosen = [entries[i] for i in pick]
    pairs = [(e["db"], e["task"]) for e in chosen]
    report["tasks"] = [f"{d}/{t}" for d, t in pairs]

    net, config = load_rt_model(ckpt, device="cuda", compile=False)
    net = net.to(torch.bfloat16).eval()
    ds = make_dataset(resolve_tasks(pre_dir, pairs), pre_dir, config, mmap_populate=True)
    batches = []
    for di, e in enumerate(chosen):
        rows = np.load(e["rows"])
        idx = np.sort(rng.choice(e["n"], size=rows_per_task, replace=False))
        b, _, _ = sample(ds, di, rows["node_idxs"][idx])
        batches.append(b)
    allb = {k: torch.cat([b[k] for b in batches], 0) for k in batches[0]}
    n = allb["node_idxs"].shape[0]
    perm = torch.from_numpy(rng.permutation(n))
    allb = {k: v[perm] for k, v in allb.items()}
    gbs = [{k: v[s : s + batch_size].cuda() for k, v in allb.items()} for s in range(0, (n // batch_size) * batch_size, batch_size)]
    report["n_batches"] = len(gbs)
    report["dtypes"] = {k: str(v.dtype) for k, v in gbs[0].items()}
    report["shapes"] = {k: list(v.shape) for k, v in gbs[0].items()}
    report["non_pad_tokens_per_row"] = float((~allb["is_padding"]).sum(1).float().mean())
    log(f"{len(gbs)} batches of {batch_size}; dtypes {report['dtypes']}")
    dump()

    def bench(name, fn, inputs, reps=reps):
        with torch.inference_mode():
            for i in inputs:
                fn(i)
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            ev0.record()
            for _ in range(reps):
                for i in inputs:
                    fn(i)
            ev1.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        k = reps * len(inputs)
        r = {"ms_wall": wall / k * 1e3, "ms_gpu_events": ev0.elapsed_time(ev1) / k,
             "rows_per_s": k * batch_size / wall}
        report.setdefault("bench", {})[name] = r
        log(f"bench {name}: {r}")
        dump()
        return r

    def prof(name, fn, inp):
        with torch.inference_mode():
            for _ in range(3):
                fn(inp)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                t = time.perf_counter()
                for _ in range(5):
                    fn(inp)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t) / 5
        ka = p.key_averages()
        rows = []
        for e in ka:
            dt = getattr(e, "self_device_time_total", None)
            if dt is None:
                dt = e.self_cuda_time_total
            if dt > 0:
                rows.append((e.key, dt / 5 / 1e3, e.count // 5))
        rows.sort(key=lambda r: -r[1])
        busy = sum(r[1] for r in rows)
        report.setdefault("profile", {})[name] = {
            "wall_ms": wall * 1e3, "gpu_busy_ms": busy,
            "top": [{"op": k[:160], "ms": ms, "calls": c} for k, ms, c in rows[:40]],
        }
        log(f"profile {name}: wall {wall*1e3:.1f} ms, gpu busy {busy:.1f} ms")
        for k, ms, c in rows[:25]:
            log(f"   {ms:8.2f} ms  x{c:<5d} {k[:120]}")
        (out / f"profile_{name}.txt").write_text(ka.table(sort_by="self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total", row_limit=60, max_name_column_width=120))
        dump()

    def full(model):
        return lambda b: model(b, return_embeddings=True)

    with torch.inference_mode():
        sframes = [sort_frame(b) for b in gbs]
        dm = [dense_masks(s) for s in sframes]
        bm_dense = [block_masks_from_dense(m, 128) for m, _ in dm]
        kv_dense = [kv for _, kv in dm]
        xs = [encode(net, s) for s in sframes]
        report["block_sparsity_128"] = block_stats(bm_dense[0])
        report["block_sparsity_64"] = block_stats(block_masks_from_mods(sframes[0], 64))
        report["block_sparsity_32"] = block_stats(block_masks_from_mods(sframes[0], 32))
        kvf = kv_sizes_fast(sframes[0])
        report["kv_sizes_fast_matches_dense"] = {k: float((kvf[k].float() - kv_dense[0][k].float()).abs().max()) for k in kvf}
        dens = {k: float(m.float().mean()) for k, m in dm[0][0].items()}
        report["mask_density"] = dens
        ref = net(gbs[0], return_embeddings=True)
        mine = run_blocks(net, xs[0], bm_dense[0], kv_dense[0])
        report["decomposed_matches_forward_max_abs"] = float((ref.float() - mine.float()).abs().max())
    log(f"sparsity {report['block_sparsity_128']} density {dens}; decomposed diff {report['decomposed_matches_forward_max_abs']}")
    dump()

    idx = list(range(len(gbs)))
    bench("eager_full_materialize", full(net), gbs)
    prof("eager_full_materialize", full(net), gbs[0])
    bench("eager_sort_gather", sort_frame, gbs)
    bench("eager_dense_masks_and_kv", dense_masks, sframes)
    bench("eager_block_mask_from_dense_x3", lambda m: block_masks_from_dense(m[0], 128), dm)
    bench("eager_block_mask_from_mods_x3", lambda s: block_masks_from_mods(s, 128), sframes)
    bench("eager_kv_sizes_fast", kv_sizes_fast, sframes)
    bench("eager_encoders", lambda s: encode(net, s), sframes)
    bench("eager_blocks_prebuilt_masks", lambda i: run_blocks(net, xs[i], bm_dense[i], kv_dense[i]), idx)
    bench("eager_blocks_dense_flex_no_mask", lambda i: run_blocks(net, xs[i], {"feat": None, "nbr": None, "col": None}, kv_dense[i]), idx)
    bench("eager_blocks_sdpa_no_mask", lambda i: run_blocks_sdpa(net, xs[i], kv_dense[i]), idx)
    prof("eager_blocks_prebuilt_masks", lambda i: run_blocks(net, xs[i], bm_dense[i], kv_dense[i]), 0)

    net.materialize_attn_masks = False
    bench("eager_full_no_materialize", full(net), gbs)
    net.materialize_attn_masks = True

    netc, _ = load_rt_model(ckpt, device="cuda", compile=True)
    netc = netc.to(torch.bfloat16).eval()
    t = time.perf_counter()
    with torch.inference_mode():
        netc(gbs[0], return_embeddings=True)
    torch.cuda.synchronize()
    report["compile_first_call_s"] = time.perf_counter() - t
    bench("compiled_full_materialize", full(netc), gbs)
    prof("compiled_full_materialize", full(netc), gbs[0])

    netnm, _ = load_rt_model(ckpt, device="cuda", compile=True, model_kwargs={"materialize_attn_masks": False})
    netnm = netnm.to(torch.bfloat16).eval()
    bench("compiled_full_no_materialize", full(netnm), gbs)
    prof("compiled_full_no_materialize", full(netnm), gbs[0])
    del netnm

    blocks_c = torch.compile(partial(run_blocks, net), dynamic=False)
    bench("compiled_blocks_prebuilt_masks", lambda i: blocks_c(xs[i], bm_dense[i], kv_dense[i]), idx)
    prof("compiled_blocks_prebuilt_masks", lambda i: blocks_c(xs[i], bm_dense[i], kv_dense[i]), 0)
    bench("compiled_blocks_dense_flex_no_mask", lambda i: blocks_c(xs[i], {"feat": None, "nbr": None, "col": None}, kv_dense[i]), idx)
    sdpa_c = torch.compile(partial(run_blocks_sdpa, net), dynamic=False)
    bench("compiled_blocks_sdpa_no_mask", lambda i: sdpa_c(xs[i], kv_dense[i]), idx)
    prep_c = torch.compile(lambda b: dense_masks(sort_frame(b)), dynamic=False)
    bench("compiled_sort_and_dense_masks", prep_c, gbs)
    enc_c = torch.compile(partial(encode, net), dynamic=False)
    bench("compiled_encoders", enc_c, sframes)
    bm64 = [block_masks_from_mods(s, 64) for s in sframes]
    kvf = [kv_sizes_fast(s) for s in sframes]
    try:
        bench("compiled_blocks_block64_masks", lambda i: blocks_c(xs[i], bm64[i], kvf[i]), idx)
    except Exception as exc:
        report["compiled_blocks_block64_error"] = repr(exc)[:2000]
        dump()

    for name, (m, k, nn_) in {"proj_512x512": (batch_size * 256, 512, 512), "ffn_512x2048": (batch_size * 256, 512, 2048)}.items():
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        wt = torch.randn(nn_, k, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            r = bench(f"gemm_{name}", lambda _: torch.nn.functional.linear(a, wt), [0])
        report["bench"][f"gemm_{name}"]["tflops"] = 2 * m * k * nn_ / (r["ms_wall"] / 1e3) / 1e12
        del a, wt

    for mode in ("max-autotune-no-cudagraphs",):
        try:
            netm, _ = load_rt_model(ckpt, device="cuda", compile=False)
            netm = netm.to(torch.bfloat16).eval()
            fm = torch.compile(netm.forward, dynamic=False, mode=mode)
            bench(f"compiled_full_materialize_{mode}", lambda b: fm(b, return_embeddings=True), gbs)
            del netm, fm
        except Exception as exc:
            report[f"mode_{mode}_error"] = repr(exc)[:2000]
            dump()

    netc.train()
    tb = train_batch_size
    tgbs = [{k: v[:tb] for k, v in b.items()} for b in gbs]
    for p in netc.parameters():
        p.requires_grad_(True)

    def step(b):
        loss = netc(b, return_embeddings=False)[0]
        loss.backward()
        netc.zero_grad(set_to_none=True)

    for b in tgbs:
        step(b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        for b in tgbs:
            step(b)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / (reps * len(tgbs))
    report["train_fwd_bwd_S256"] = {"batch": tb, "ms": wall * 1e3, "seqs_per_s": tb / wall, "tokens_per_s": tb * 256 / wall}
    log(f"train fwd+bwd: {report['train_fwd_bwd_S256']}")
    dump()
    log("done")
