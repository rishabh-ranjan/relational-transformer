from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR

# The Join, preprocessed: the corpus RT-J was pretrained on. Goes beside the
# other stanford-star data so it is a pre_dir every node can read.
#
# docs/downloads.md says ~1.5 TiB and names nodes.rkyv at ~1055 GiB. That is
# stale. The Hub reports 4189 files and 0.25 TiB logical -- nodes.rkyv 195.1,
# text_emb_all-MiniLM-L12-v2.bin 29.8, p2f_adj.rkyv 26.1, offsets.rkyv 5.3,
# text.json 3.2 GiB. Against the compression /dfs gets on the same file types
# in relbench-preprocessed (nodes.rkyv 6.3x, p2f_adj 3.1x, offsets 1.7x,
# text.json 1.7x, the float32 embeddings 1.0x) that is ~75 GiB on disk,
# against 876 GiB free.
#
# Everything is fetched, text.json included: the docs' subset recipe exists to
# dodge a 27 GiB file that is actually 3.2, so the complication no longer pays
# for itself.
REPO = "stanford-star/the-join-preprocessed"
DEST = "~/scratch/hf/stanford-star/the-join-preprocessed"
REPO_ROOT = str(Path(__file__).resolve().parents[4])

submit(
    "expts.repaper.baselines.fetch_the_join:main",
    args=dict(
        repo=REPO,
        dest=DEST,
        max_workers=16,
        allow_patterns=None,
    ),
    resources=Resources(
        # Zero-gres, so il-cpu: uncapped, 31-day wall, and it keeps a
        # many-hour download off the gpu nodes entirely. furiosa is an il-cpu
        # node that is already set up.
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time="24:00:00",
        gpus="0",
        cpus_per_task=16,
        ntasks=1,
        exclusive=False,
        mem="32G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="furiosa",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name="fetch-the-join-preprocessed",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
