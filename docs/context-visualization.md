# Context visualization

RT predicts from the **context** sampled for each row, so it helps to *see* what
a given context config actually pulls in. `ctx-viz` serves an interactive HTTP UI
over preprocessed data: pick a dataset/task and a seed row, and inspect the
sampled context (the neighborhood of cells the model attends over).

```bash
pixi run ctx-viz --pre-root stanford-star/relbench-preprocessed   # Hub repo
pixi run ctx-viz --pre-root ~/scratch/pre                        # or a local root
```

Then open the printed URL. It works against both local and Hub preprocessed data
(same local-or-Hub interface as everything else — see
[preprocess.md](preprocess.md)). This is the tool to reach for when
[context-engineering](inference.md#context-engineering) a task: tweak
`local_ctx_size` / `bfs_width` and watch how the sampled context changes.
