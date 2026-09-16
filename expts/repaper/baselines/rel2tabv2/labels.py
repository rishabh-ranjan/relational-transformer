import torch


class PreprocessedLabels:
    def __init__(self, pre_dir, embedder, d_text):
        self.pre_dir = pre_dir
        self.embedder = embedder
        self.d_text = d_text
        self._samplers = {}

    def _sampler(self, task):
        key = (task.db_name, task.table_name, task.split)
        if key not in self._samplers:
            from rt.data import RustlerDataset

            self._samplers[key] = RustlerDataset(
                tasks=[task],
                pre_dir=self.pre_dir,
                global_rank=0,
                local_rank=0,
                world_size=1,
                local_ctx_size_list=[1],
                bfs_width_list=[0],
                num_walks=0,
                walk_length=0,
                prefer_latest_list=[False],
                mask_prob_max=0.0,
                embedder=self.embedder,
                d_text=self.d_text,
                shuffle_seed=0,
                context_seed=0,
                items_per_task=10_000_000,
                quiet=True,
                ignore_data_errors=False,
                mmap_populate=True,
                timeout_per_item=3600.0,
                vector_db_path=None,
                db_cutoff=None,
                legacy_boolean=False,
            )
        return self._samplers[key]

    def labels_for(self, task, node_idxs, batch_size=4096):
        # The evaluator's labels are the preprocessed target cell: a z-scored
        # boolean for clf, which the predictors' `> 0` test binarises correctly,
        # and a normalised target for reg, which is the space rt.eval.relbench
        # denormalises out of. Reading that same cell is what makes one label
        # source correct for both, with no task-type branch.
        from rt.data import process_batch

        ds = self._sampler(task)
        out = torch.empty(len(node_idxs), dtype=torch.float32)
        for start in range(0, len(node_idxs), batch_size):
            chunk = [int(v) for v in node_idxs[start : start + batch_size]]
            # ctx_size=1: seq_build pushes the target cell first, so the
            # sequence is exactly that cell and nothing else.
            tup = ds.sampler.batch_for_nodes_py(chunk, 0, 1)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            is_targets = batch["is_targets"]
            per_row = is_targets.sum(dim=1)
            assert (per_row == 1).all(), (
                f"{task.db_name}/{task.table_name}: expected one target cell "
                f"per row, got counts {per_row.unique().tolist()}"
            )
            vals = batch["number_values"].squeeze(-1)
            out[start : start + len(chunk)] = (vals * is_targets.to(vals.dtype)).sum(
                dim=1
            )
        return out
