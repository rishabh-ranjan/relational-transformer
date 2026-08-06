"""Evaluator: the shared per-(task, ctx_size) eval kernel used by standalone
eval and by in-loop training eval (rank-aware under DDP)."""

import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from rt.data import EvalDataset, RustlerDataset
from rt.progress import fmt_duration, log
from rt.model.net import SEM_TYPE_BOOLEAN


class Evaluator:
    """Standard per-task eval over a fixed task list.

    Build once with sampler/loader knobs; call ``evaluate_raw`` one or
    more times. Re-using an instance across
    eval points reuses prefetch state and avoids loader rebuild —
    important for the in-loop training eval.

    Synthetic-DB tasks (``"synthetic" in db_name``) are dropped at
    construction; they were never evaluated by the inline path either.
    """

    def __init__(
        self,
        *,
        tasks,
        pre_dir,
        eval_bs,
        ctx_size_list,
        items_per_task,
        num_workers,
        prefetch_factor,
        persistent_workers,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        mmap_populate,
        embedder,
        d_text,
        shuffle_seed,
        context_seed,
        vector_db_path,
        train_only_fallback,
        global_rank,
        local_rank,
        world_size,
        ddp,
        device,
    ):
        self.tasks = [t for t in tasks if "synthetic" not in t.db_name]
        self.eval_splits = sorted(set(t.split for t in self.tasks if t.split))
        self.ctx_size_list = ctx_size_list
        self.eval_bs = eval_bs
        self.items_per_task = items_per_task
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.ddp = ddp
        self.device = device

        max_eval_ctx_size = max(ctx_size_list)

        self.eval_loaders = {}
        self.eval_loader_iters = {}

        init_tic = time.time()
        prefetch_time = 0.0

        for eval_task in self.tasks:
            rustler_dataset = RustlerDataset(
                tasks=[eval_task],
                pre_dir=pre_dir,
                global_rank=global_rank,
                local_rank=local_rank,
                world_size=world_size,
                local_ctx_size_list=[local_ctx_size],
                bfs_width_list=[bfs_width],
                num_walks=num_walks,
                walk_length=walk_length,
                prefer_latest_list=[prefer_latest],
                mask_prob_max=0.0,
                embedder=embedder,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=items_per_task,
                quiet=True,
                ignore_data_errors=False,
                mmap_populate=mmap_populate,
                timeout_per_item=3600.0,
                vector_db_path=vector_db_path,
                train_only_fallback=train_only_fallback,
            )
            eval_dataset = EvalDataset(
                rustler_dataset=rustler_dataset,
                eval_bs=eval_bs,
                eval_ctx_size=max_eval_ctx_size,
            )
            self.eval_loaders[eval_task] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                pin_memory=True,
                # in_order=True guarantees sampler-order yields; with False the
                # row order is worker-completion order — a timing race that breaks
                # cross-seed prediction averaging (context ensembling) on rows
                # written by eval_grid.
                in_order=True,
            )
            _prefetch_tic = time.time()
            self.eval_loader_iters[eval_task] = iter(self.eval_loaders[eval_task])
            prefetch_time += time.time() - _prefetch_tic
        if local_rank == 0:
            log(
                eval_tasks_loaded=len(self.tasks),
                elapsed=fmt_duration(time.time() - init_tic),
                prefetch_time=fmt_duration(prefetch_time),
            )

    def evaluate_raw(
        self, nets_with_prefix, eval_ctx_size_list_to_use, with_node_idxs=False
    ):
        """Per-task pipeline primitive.

        Drives per-batch forward + DDP gather + ``batch_mask`` filtering
        for every task. Yields one tuple per ``(task, ctx_size)`` on
        rank 0::

            (task, ctx_size, labels_np, preds_by_prefix_np, num_labels_np)

        - ``labels_np``: ``(n_real,)`` per-row labels.
        - ``preds_by_prefix_np``: dict ``prefix → (n_real,) preds``,
          one entry per ``(net, prefix)`` in ``nets_with_prefix``.
        - ``num_labels_np``: ``(n_real,) int64`` per-row count of
          in-context training labels for that row's target column at
          ``ctx_size`` (the ``mean_labels`` source data).

        ``n_real`` is the number of real (non-phantom) rows across all
        ranks for that task — already filtered by ``batch_mask``.

        With ``with_node_idxs=True`` a sixth element ``node_idxs_np`` is
        appended to the yielded tuple: the ``(n_real,) int64`` global
        rustler node index of each row's *seed* (target) node. Because
        rustler assigns a task row the node index ``node_idx_offset + r``
        (``r`` the 0-based row index in the relbench task-table parquet),
        ``node_idx - node_idx_offset`` recovers the exact parquet row, which
        is how :mod:`rt.eval` keys predictions back to the relbench
        ``(entity_col, time_col)`` for a leaderboard submission (eval row
        order is *not* the parquet row order, so a positional join is wrong).

        Other ranks drive every collective but yield nothing.
        """
        device = self.device
        ddp = self.ddp
        world_size = self.world_size
        global_rank = self.global_rank

        for net, _ in nets_with_prefix:
            net.eval()

        with torch.inference_mode():
            for eval_task, eval_loader_iter in self.eval_loader_iters.items():
                eval_loader = self.eval_loaders[eval_task]

                # The number of eval batches per task MUST be identical on every
                # rank, or NCCL deadlocks (ranks issue a different number of
                # collective calls). ``len(eval_loader.dataset)`` is
                # ``ceil(num_items / (eval_bs * world_size))`` -- uniform across
                # ranks (``num_items`` is the task's total item count, not a
                # per-rank count), and the rustler sampler fills any overshoot
                # slots as phantoms (batch_mask[i]=False). ``items_per_task``
                # only caps how many batches we bother running; the cap is the
                # same integer on every rank, so it never desyncs the schedule.
                n_batches = len(eval_loader.dataset)
                if self.items_per_task is not None:
                    n_batches = min(
                        n_batches,
                        max(1, self.items_per_task // self.eval_bs // world_size),
                    )

                preds_per_prefix_per_ctx = {
                    prefix: {ctx: [] for ctx in eval_ctx_size_list_to_use}
                    for _, prefix in nets_with_prefix
                }
                num_labels_per_ctx = {ctx: [] for ctx in eval_ctx_size_list_to_use}
                labels = []
                batch_masks = []
                node_idxs_acc = []
                # Drive the loop by the fixed, cross-rank-uniform batch count.
                # Every rank processes exactly ``n_batches`` batches (each of
                # ``eval_bs`` rows, phantom-padded as needed), so every rank
                # contributes exactly ``n_batches * eval_bs`` rows to every
                # collective below -- no StopIteration / local-count breaks.
                for _ in range(n_batches):
                    batch = next(eval_loader_iter)

                    batch_mask = batch.pop("batch_mask")

                    # Per-row in-context training-label count for the
                    # target column, for each requested ctx_size. Gathered
                    # and masked alongside labels/preds so the eventual
                    # mean_labels stat is uniform over real items.
                    for eval_ctx_size in eval_ctx_size_list_to_use:
                        tb = {k: v[:, :eval_ctx_size] for k, v in batch.items()}
                        tb_is_targets = tb["is_targets"]
                        tb_target_col = torch.full(
                            (tb_is_targets.shape[0], 1),
                            -1,
                            dtype=tb["col_name_idxs"].dtype,
                        )
                        tb_target_node = tb_target_col.clone()
                        tb_bidxs, tb_sidxs = tb_is_targets.nonzero(as_tuple=True)
                        tb_target_col[tb_bidxs, 0] = tb["col_name_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        tb_target_node[tb_bidxs, 0] = tb["node_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        is_label_cell = (
                            tb["is_task_nodes"]
                            & ~tb["is_padding"]
                            & (tb["col_name_idxs"] == tb_target_col)
                            & (tb["node_idxs"] != tb_target_node)
                        )
                        num_labels_per_ctx[eval_ctx_size].append(
                            is_label_cell.sum(dim=1).to(torch.int64)
                        )

                    for net, prefix in nets_with_prefix:
                        preds_by_ctx = net.predict(
                            batch,
                            eval_ctx_size_list_to_use,
                            device,
                            eval_task,
                        )
                        for ctx_size, yhat in preds_by_ctx.items():
                            assert yhat.size(0) == batch_mask.size(0)
                            preds_per_prefix_per_ctx[prefix][ctx_size].append(yhat)

                    # Read the label from the channel its semantic type names:
                    # Boolean-typed targets (legacy-preprocessed data) live in
                    # boolean_values, everything else in number_values. The
                    # stored value is the same either way, so this needs no
                    # knowledge of which net is being evaluated.
                    vals = torch.where(
                        (batch["sem_types"] == SEM_TYPE_BOOLEAN).unsqueeze(-1),
                        batch["boolean_values"],
                        batch["number_values"],
                    ).squeeze(-1)
                    y = (vals * batch["is_targets"].to(vals.dtype)).sum(dim=1)
                    assert y.size(0) == batch_mask.size(0)
                    labels.append(y)
                    batch_masks.append(batch_mask)
                    if with_node_idxs:
                        # Seed (target) node's global rustler index per row. Exactly
                        # one target cell per real row, so the masked sum picks it
                        # out; phantom rows have no target → 0, dropped by batch_mask.
                        nidx = (
                            batch["node_idxs"].to(torch.int64)
                            * batch["is_targets"].to(torch.int64)
                        ).sum(dim=1)
                        assert nidx.size(0) == batch_mask.size(0)
                        node_idxs_acc.append(nidx)

                # prefetch next pass while we run gather + metric compute.
                self.eval_loader_iters[eval_task] = iter(eval_loader)

                # Every rank ran exactly ``n_batches`` batches of ``eval_bs``
                # rows, so ``labels_cat`` has the same length on every rank and
                # the all_gathers are inherently in lockstep -- no cross-rank
                # MIN reduce or truncation needed. Phantom rows are filtered out
                # via ``masks_gathered`` on rank 0 after the gather.
                labels_cat = torch.cat(labels, dim=0).to(device)
                masks_cat = torch.cat(batch_masks, dim=0).to(device)
                if ddp:
                    labels_gathered = torch.empty(
                        labels_cat.size(0) * world_size,
                        dtype=labels_cat.dtype,
                        device=device,
                    )
                    masks_gathered = torch.empty(
                        masks_cat.size(0) * world_size,
                        dtype=masks_cat.dtype,
                        device=device,
                    )
                    dist.all_gather_into_tensor(
                        labels_gathered, labels_cat.contiguous()
                    )
                    dist.all_gather_into_tensor(masks_gathered, masks_cat.contiguous())
                else:
                    labels_gathered = labels_cat
                    masks_gathered = masks_cat

                if global_rank == 0:
                    labels_np = labels_gathered[masks_gathered].float().cpu().numpy()

                node_idxs_np = None
                if with_node_idxs:
                    nidx_cat = torch.cat(node_idxs_acc, dim=0).to(device)
                    if ddp:
                        nidx_gathered = torch.empty(
                            nidx_cat.size(0) * world_size,
                            dtype=nidx_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nidx_gathered, nidx_cat.contiguous()
                        )
                    else:
                        nidx_gathered = nidx_cat
                    if global_rank == 0:
                        node_idxs_np = nidx_gathered[masks_gathered].cpu().numpy()

                for eval_ctx_size in eval_ctx_size_list_to_use:
                    nlabels_cat = torch.cat(
                        num_labels_per_ctx[eval_ctx_size], dim=0
                    ).to(device)
                    if ddp:
                        nlabels_gathered = torch.empty(
                            nlabels_cat.size(0) * world_size,
                            dtype=nlabels_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nlabels_gathered, nlabels_cat.contiguous()
                        )
                    else:
                        nlabels_gathered = nlabels_cat

                    preds_by_prefix_np = {}
                    for _, prefix in nets_with_prefix:
                        preds = torch.cat(
                            preds_per_prefix_per_ctx[prefix][eval_ctx_size], dim=0
                        ).to(device)
                        if ddp:
                            preds_gathered = torch.empty(
                                preds.size(0) * world_size,
                                dtype=preds.dtype,
                                device=device,
                            )
                            dist.all_gather_into_tensor(
                                preds_gathered, preds.contiguous()
                            )
                            preds = preds_gathered
                        if global_rank == 0:
                            preds_by_prefix_np[prefix] = (
                                preds[masks_gathered].float().cpu().numpy()
                            )

                    if global_rank == 0:
                        num_labels_np = nlabels_gathered[masks_gathered].cpu().numpy()
                        out = (
                            eval_task,
                            eval_ctx_size,
                            labels_np,
                            preds_by_prefix_np,
                            num_labels_np,
                        )
                        if with_node_idxs:
                            out = out + (node_idxs_np,)
                        yield out
