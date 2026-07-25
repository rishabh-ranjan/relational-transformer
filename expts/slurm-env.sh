# Shared environment for slurm jobs in expts/ -- source, don't execute.
#
# A batch script is not a login shell, so none of the interactive fish config
# runs. This file is the single place that defines what a job's env looks like,
# so no individual job script has to get it right. It is node-agnostic: every
# path is derived on the machine it runs on, so it behaves the same on the dev
# node and on any compute node.
#
# No fallbacks: anything missing is a hard error here, seconds into the job,
# rather than a silently degraded run discovered hours later.

set -euo pipefail

die() { echo "slurm-env: $*" >&2; exit 1; }

export USER=${USER:-$(id -un)}

# Node-local home: pixi envs, HF/torch caches, wandb staging.
export HOME=/lfs/local/0/$USER
mkdir -p "$HOME" || die "no node-local scratch at $HOME on $(hostname -s)"
export PATH=$HOME/.pixi/bin:/sailhome/$USER/.pixi/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # a dev-node-local dir; would shadow the pixi env
export TMPDIR=/tmp/$USER
export XDG_CACHE_HOME=$HOME/.cache
export WANDB_DIR=$HOME/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# pixi is a single static binary in the shared home, so it exists on every node
# even when this node has never been used before; the node-local copy just wins
# on PATH when present. The package cache is node-local, and warm across jobs
# that land on the same node.
command -v pixi >/dev/null || die "pixi not on PATH ($PATH)"
export PIXI_CACHE_DIR=$HOME/.cache/rattler/cache

# Tokens live in the shared secrets dir (readable from every node) rather than
# being exported into the job env, where slurm would record them.
_secrets=/sailhome/$USER/.secrets
_read_secret() {
    [[ -r $_secrets/$1 ]] || die "missing secret $_secrets/$1"
    local v
    v=$(tr -d '[:space:]' < "$_secrets/$1")
    [[ -n $v ]] || die "empty secret $_secrets/$1"
    printf '%s' "$v"
}
WANDB_API_KEY=$(_read_secret wandb); export WANDB_API_KEY
HF_TOKEN=$(_read_secret huggingface); export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
GITHUB_TOKEN=$(_read_secret github); export GITHUB_TOKEN GH_TOKEN=$GITHUB_TOKEN

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
ulimit -l unlimited || die "cannot raise RLIMIT_MEMLOCK (need --propagate=MEMLOCK)"

echo "slurm-env: host=$(hostname -s) HOME=$HOME pixi=$(command -v pixi)"
