from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
    project,
)

# The batch, bought with hardware the way pretraining bought it. rt-j averaged
# total_bs=1024 independently drawn items per step over 8 a100s under DDP; the
# single-gpu adapter runs averaged 4 task draws, and rows sharing a context are
# strongly correlated, so 4 x 256 query rows is 4 samples against between-task
# variance -- the noise that made v1's loss curve unreadable.
#
# 4 ranks x 32 accumulation micro-steps x 4 tasks = 512 tasks per optimiser
# step, 128x the single-gpu run.
#
# COST: ~0.55 s per task draw measured on v1/v3 (2.19-2.30 s/step at 4 tasks),
# so 128 tasks per rank per step is ~70 s/step. 10,000 steps would be ~195 h
# = 8.1 days, over il's 7-day cap, so this runs 2,500 -- still 1.28M task
# draws, 32x the 40k of the whole v1 run, and it completes rather than being
# cut off mid-schedule. 8 ranks would fit 10k in ~4.1 days if this one says a
# bigger batch is what was missing.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

# The fp32 arm is job 191576, already running on ampere3 -- do not resubmit it.
# This is the bf16 arm: same seed, so it draws the *identical* task sequence,
# and precision is the only difference. A subagent measured the fp32 run at
# 97-98% gpu util, 5.71 TFLOP/s on 3.2 TFLOP/draw = 29% of the a100's 19.5
# TFLOP/s fp32 non-tensor-core peak, with TF32 off (allow_tf32 False,
# float32_matmul_precision "highest") -- so it is compute-bound on the wrong
# execution units. bf16 puts matmuls on tensor cores and lets sdpa select
# FlashAttention. Expected 2-3x, shape-limited rather than FLOP-limited
# (attention is (1, 554, 16, 64): batch 1, 16 heads over 108 SMs).
# v6. Three changes from v4/v5, all aimed at the shape problem:
#   d_out 64      TabPFN is a *cell*-level transformer: cost scales with the
#                 feature axis, and 512 features x ~810 rows is ~415k cell
#                 positions for ONE draw -- about 3x rt-j pretraining's entire
#                 131k-token micro-batch. Running a 256-row context against a
#                 512-dim vector is also a degenerate regime for TabPFN
#                 (rows < features). 64 also lands entirely inside
#                 max_features_per_estimator=500, so the 12-13 permanently
#                 dead output dims disappear.
#   n_ctx ladder  one rung per micro-step shared by every draw in it, seeded
#                 off (seed, step, micro) and NOT off rank so the ranks stay
#                 in lockstep -- rt-j pretraining's scheme (datasets.py:279).
#                 This is the precondition for stacking draws into one forward.
#   bf16          forced via inference_precision, i.e. cast the model once as
#                 pretraining does, NOT autocast. Autocast made the adapter
#                 emit bf16 into AddFingerprintFeaturesStep, which hashes via
#                 .numpy() and has no bf16 -- that killed v5 at step 0. Forced
#                 bf16 casts in _prepare_model_inputs (inference.py:1308)
#                 *after* preprocessing, so the fingerprint step still sees
#                 fp32. Halves activations, which is what buys batch size.
#
# NOTE draws are still processed one at a time. The ladder makes stacking
# possible; the stacking itself needs fit_from_preprocessed and is not in yet.
# SMOKE: 1 gpu, 5 steps, no eval, no wandb -- measure s/draw at d_out=64
# + bf16 + ladder before spending a 4-gpu slot.
# fp32: bf16 needs FINGERPRINT_FEATURE off (the torch fingerprint step in
# the GPU pipeline hashes via .numpy() AFTER the bf16 cast at
# inference.py:1308), and that changes the model inputs. Measure d_out=64
# on its own first -- if it drops peak memory enough, bf16 is not needed.
# Smoke (job 191935, 1 gpu) cleared the fused path: both task types ran,
# gradient nonzero, 0 dropped, 0 degenerate, peak 18.0 GiB, and
# 0.0938 s/draw against production 0.5581 = 6.0x.
# v8: v7 for 10k steps with a cosine schedule, peak 5e-4 -> 1e-5.
# 5e-4 is pretraining's lr and 5x v7's; Adam's step is ~lr regardless of
# batch, so 2500 steps at 1e-4 moves each weight at most ~0.25 in total.
# With the gradient now at SNR ~0.95 rather than 0.084, bigger steps in a
# trustworthy direction are the point.
# v9: v8 with the cosine peak at 1e-3 instead of 5e-4. Adam's step is ~lr
# regardless of batch size, so peak lr is what sets how far the adapter can
# actually travel; v7 at a flat 1e-4 moves each weight at most ~0.25 over
# 2500 steps. The gradient is now at SNR ~0.95 (B_simple = (18/0.76)^2 =
# 561, so 512 draws is the canonical one-noise-scale operating point), so
# the direction is worth taking big steps in. Runs alongside v8 as an lr
# sweep; everything else is identical.
# v10: v9 plus an EMA of the head weights, evaluated and checkpointed
# alongside the live iterate. Nothing about training changes.
# The trainer now resumes: resume.pt every resume_save_mins and on
# SIGTERM/SIGUSR1, reloaded at startup from the same out_dir. Smoked on one
# gpu at 30 steps with wandb on and eval_every 10 (192902 uninterrupted,
# 192924 stopped with USR1 at step 14, 192932 and 192937 resumed at step 15
# with swa n and the AdamW moments intact). il-lo is therefore open; the
# qos below is the only line that has to change for it.
# v10: v9's recipe with a SwiGLU adapter instead of a linear map --
# 512 -> 64 -> 64, rt-j's own FFN shape (net.py:88-99). On il-lo, which is
# only safe now that the trainer resumes from resume.pt after a preempt.
# v11: v10 with a bias on the two SwiGLU INPUT projections. The output
# bias stays off because it is provably inert -- TabPFN z-norms each
# column against the context and subtracts it back off, zero gradient --
# but w1/w3 feed a silu gate and a multiply, so a bias there shifts each
# hidden unit along the gate curve and does change the function.
ARMS = [("bf16", "join-v11-ddp-swiglu64-inbias")]
# ARMS = [("bf16", "join-v10-ddp-swiglu64")]
# ARMS = [("bf16", "join-v9-ddp-proj64-cosine1e-3")]
# ARMS = [("bf16", "join-v9-ddp-proj64-cosine1e-3")]
# ARMS = [("bf16", "join-v8-ddp-proj64-cosine")]
# ARMS = [("bf16", "join-v7-ddp-proj64-batched")]
# ARMS = [(None, "join-v4-ddp-linear")]

for autocast, RUN in ARMS:

    submit(
        "expts.repaper.adapter.train_adapter_ddp:main",
        args=dict(
            features_root=f"{SHARE}/features_join_u12",
            relbench_pre_dir=PRE_DIR,
            relbench_features_root=f"{SHARE}/features_rt-j",
            relbench_labels_root=f"{SHARE}/labels_relbench",
            relbench_n_ctx=8192,
            relbench_n_query=4096,
            tabpfn_dir=f"{SHARE}/tabpfn",
            stats_path=f"{SHARE}/feature_stats_join_u12.npz",
            out_dir=f"{OUT_ROOT}/adapter/{RUN}",
            d_feat=512,
            d_out=64,
            min_rows=512,
            n_ctx_list=[256, 512, 1024],
            n_query=256,
            # 4 ranks x 4 tasks x 32 accum. The script asserts the divisibility, so
            # changing the rank count without changing this fails at startup rather
            # than silently running a different batch.
            total_tasks_per_step=512,
            # Aligned with n_ctx_list. Chosen so peak memory is roughly flat
            # across rungs and total_tasks_per_step divides exactly at every
            # one. Measured frontier at d_out=64/bf16: s/draw is flat past
            # B~8-12, so 32/16/8 captures essentially all of it while leaving
            # over half the card free for the relbench eval draw.
            micro_batch_list=[32, 16, 8],
            total_steps=10_000,
            lr=1e-3,
            lr_min=1e-5,
            # Zero, deliberately. AdamW's decay pulls a weight toward 0, and this
            # weight starts at I -- decaying it is decaying the rt-j featurizer
            # away, not regularising the adapter toward it.
            wd=0.0,
            warmup_steps=100,
            # 10.0, the middle arm of the single-gpu clip sweep. A 512-task
            # gradient should also be far steadier than a 4-task one, so this
            # ought to bind on a small minority of steps; train/frac_clipped says.
            grad_norm_max=10.0,
            # rt-j pretraining's value (pretrain/submit_ilc.py:57). The
            # gradient here has SNR ~0.95 per step at 512 draws, so the iterate
            # sits in a noise ball; Polyak-Ruppert averaging is the standard
            # answer. 1/(1-m) = 2000 steps of effective window against a 10k
            # run. Well-posed: the objective's only gauge freedom is per-row
            # positive rescaling of the head, that direction has exactly zero
            # gradient, and wd=0.0, so nothing pushes iterates into different
            # gauges.
            swa_momentum=0.9995,
            precision=autocast,
            adapter_kind="swiglu_bias",
            hidden_dim=64,
            # At ~70 s/step these are ~30 min and ~1 h of wall clock, not the
            # 10 min and 20 min they were on one gpu.
            # 4x the steps, so eval and checkpoint cadence scale with it:
            # eval is 76.8 s and was 36% of v7's wall clock, and 400 evals
            # over 10k steps would cost 8.5 h of the ~17 h run.
            eval_every=100,
            save_every=200,
            # rt-j pretraining's cadence (pretrain/submit_ilc.py:65). Time and
            # not steps because preemption is a wall-clock event: what a
            # requeue costs is the minutes since the last write, and on il-lo
            # slurm gives 300 s of grace, so at worst 20 min is redone.
            resume_save_mins=20.0,
            seed=0,
            targets={"val/auroc": 0.7173, "val/nmae": 0.3584},
            run_name=f"adapter-{RUN}",
            project=project("adapter"),
            entity="rtv2",
            wandb_disabled=False,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            # Now that the run writes resume.pt and picks it back up, il-lo is
            # open: uncapped instead of 10 a100s, 21 d instead of 7, at the
            # price of a requeue that costs the minutes since the last save.
            qos="il",
            # qos="il-lo",
            # ~49 h at 70 s/step; 72 leaves room for a slow node.
            # ~8.3 h at 0.0938 s/draw; 24 leaves room for a slow node.
        # ~15 h train + ~2.1 h eval; 48 h leaves room for a slow node.
        time="2-00:00:00",
            # One rank per gpu: roach maps SLURM_PROCID -> RANK and SLURM_NTASKS ->
            # WORLD_SIZE (roach/slurm/run.py:19), so ntasks=None gives 4 ranks.
            gpus="a100:4",
            cpus_per_task=14,
            ntasks=None,
            exclusive=False,
            mem="240G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere6,ampere7,ampere9",
        ),
        name=f"adapter-train-{RUN}",
        run_id=None,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
