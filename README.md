# Relational Transformer (RT)

This repository is the official implementation
of the Relational Transformer (RT) architecture
for building Relational Foundation Models (RFMs).

| Paper | Venue | Implementation |
|---|---|---|
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | [`stanford-star/plurel`](https://github.com/stanford-star/plurel) |
| RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions | In progress | [`main`](https://github.com/stanford-star/relational-transformer) |



## Quickstart

### Get started with Colab

Try RT without any local setup: the [fully worked Colab
notebook](examples/byod/colab.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/examples/byod/colab.ipynb))
predicts on a general database (which could be your own!)
with a released RT-J checkpoint
end-to-end.
The same flow as plain scripts — with the checkpoint picked straight
from the Hugging Face Hub — is in
[examples/inference](examples/inference/README.md).

### Install locally

We use [pixi](https://pixi.sh) to manage a single, self-contained environment
(Python, PyTorch + CUDA, Rust, and other dependencies).
All commands are run using `pixi run` to use the environment.
Pixi builds the environment automatically on first use
(check out the docs below).

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
```

| Docs | Description |
|---|---|
| [Downloads](docs/downloads.md) | Bulk-download raw data, preprocessed data, and checkpoints from our HuggingFace org |
| [Preprocess](docs/preprocess.md) | Convert RelBench-format databases into RT's on-disk format |
| [Inference](docs/inference.md) | Run a trained checkpoint; evaluate, engineer, tune, and ensemble contexts |
| [Pretrain](docs/pretrain.md) | Train RT from scratch, single-GPU to multi-node |
| [Baselines](docs/baselines.md) | rel2tab tabular baselines through the same eval path |
| [Context visualization](docs/context-visualization.md) | Inspect the contexts sampled for each row |
