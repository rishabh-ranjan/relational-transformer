RUN_TAG = "2026-08-19"

CKPT_CLF = "~/scratch/hf/stanford-star/rt-j/classification"
CKPT_REG = "~/scratch/hf/stanford-star/rt-j/regression"

PRE_DIR = "~/scratch/hf/stanford-star/relbench-preprocessed"
RAW_DIR = "~/scratch/hf/stanford-star/relbench"

SHARE = "~/scratch/hf/relational-transformer/repaper"

CKPT_ROOT = "~/scratch/ckpts"
OUT_ROOT = f"{CKPT_ROOT}/rtv2"
LOG_ROOT = "~/scratch/relational-transformer"
CLONE_ROOT = "~/roach_clones"
SECRETS_DIR = "~/scratch/.secrets"


def project(name: str) -> str:
    return f"{RUN_TAG}-repaper-{name}"
