"""Measure the embedding stage: 1 GPU vs every GPU on the node, same texts.

Run as a roach target so it gets the project's environment and a whole node's
GPUs in one rank.
"""

from __future__ import annotations

import time


def bench(*, text_path: str, n_texts: int, batch_size: int, embedder: str) -> None:
    import orjson
    import torch

    from rt.preprocess.embed import TextEmbedder

    with open(text_path, "rb") as f:
        texts = orjson.loads(f.read())
    texts = texts[:n_texts]
    ngpu = torch.cuda.device_count()
    card = torch.cuda.get_device_name(0)
    print(
        f"host={__import__('socket').gethostname()} gpus={ngpu} card={card}", flush=True
    )
    print(f"texts={len(texts):,} batch={batch_size}", flush=True)

    # warm the model in, so the first timing is not the download/load
    TextEmbedder(batch_size, embedder, "cuda:0")

    one = TextEmbedder(batch_size, embedder, "cuda:0")
    t0 = time.monotonic()
    one(texts, device="cuda:0")
    single = time.monotonic() - t0
    print(
        f"RESULT 1gpu   {single:8.1f}s  {len(texts) / single:9.0f} texts/s", flush=True
    )

    if ngpu > 1:
        devs = [f"cuda:{i}" for i in range(ngpu)]
        many = TextEmbedder(batch_size, embedder, devs[0])
        t0 = time.monotonic()
        many(texts, device=devs)
        multi = time.monotonic() - t0
        print(
            f"RESULT {ngpu}gpu  {multi:8.1f}s  {len(texts) / multi:9.0f} texts/s"
            f"  speedup {single / multi:.2f}x  efficiency {100 * single / multi / ngpu:.0f}%",
            flush=True,
        )
