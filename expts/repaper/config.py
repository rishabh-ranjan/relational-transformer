RUN_TAG = "2026-08-19"

CKPT_CLF = "~/scratch/share/stanford-star/rt-j/classification"
CKPT_REG = "~/scratch/share/stanford-star/rt-j/regression"

PRE_DIR = "~/scratch/share/stanford-star/relbench-preprocessed"
RAW_DIR = "~/scratch/share/stanford-star/relbench"
JOIN_PRE_DIR = "~/scratch/share/stanford-star/the-join-preprocessed"

SHARE = "~/scratch/share/relational-transformer/repaper"

CKPT_ROOT = "~/scratch/ckpts"
OUT_ROOT = f"{CKPT_ROOT}/rtv2"
LOG_ROOT = "~/scratch/relational-transformer"
CLONE_ROOT = "~/roach_clones"
SECRETS_DIR = "~/scratch/.secrets"


def project(name: str) -> str:
    return f"{RUN_TAG}-repaper-{name}"
