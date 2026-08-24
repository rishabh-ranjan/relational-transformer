import time
import orjson
import numpy as np
import torch
from rt.preprocess.embed import TextEmbedder


def _texts(text_path: str, n: int) -> list[str]:
    with open(text_path, "rb") as f:
        return orjson.loads(f.read())[:n]


def dtypes(*, text_path: str, n_texts: int, batch_size: int, embedder: str) -> None:
    texts = _texts(text_path, n_texts)
    ngpu = torch.cuda.device_count()
    one = TextEmbedder(batch_size, embedder, "cuda:0")
    p = next(one.model.parameters())
    print(f"model dtype on one device: {p.dtype}", flush=True)
    a = one(texts, device="cuda:0")

    devs = [f"cuda:{i}" for i in range(ngpu)]
    many = TextEmbedder(batch_size, embedder, devs[0])
    b = many(texts, device=devs)
    print(f"stored dtypes: single={a.dtype} multi={b.dtype}", flush=True)

    af, bf = a.astype(np.float32), b.astype(np.float32)
    same = np.array_equal(af, bf)
    print(
        f"RESULT identical={same}  max|diff|={np.abs(af - bf).max():.3e}  "
        f"rows={len(texts)}",
        flush=True,
    )


def bench(*, text_path: str, n_texts: int, batch_size: int, embedder: str) -> None:
    texts = _texts(text_path, n_texts)
    ngpu = torch.cuda.device_count()
    print(
        f"gpus={ngpu} card={torch.cuda.get_device_name(0)} texts={len(texts):,}",
        flush=True,
    )
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
