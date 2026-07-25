# Shared environment for slurm jobs in expts/ -- source, don't execute.
#
# A batch script is not a login shell, so none of the interactive fish config
# runs. This file is the single place that defines what a job's env looks like,
# so no individual job script has to get it right.
#
# It assumes nothing about the node it lands on: everything durable (the pixi
# install, the API tokens) comes from /dfs/user/$USER, which every node mounts,
# and everything node-local (home, caches, tmp) is created on demand.
#
# No fallbacks: anything missing is a hard error here, seconds into the job,
# rather than a silently degraded run discovered hours later.

set -euo pipefail

die() { echo "slurm-env: $*" >&2; exit 1; }

export USER=${USER:-$(id -un)}
SHARED=/dfs/user/$USER
[[ -d $SHARED ]] || die "$SHARED not mounted on $(hostname -s)"

# Node-local home: caches, and scratch for anything that writes to ~. Created
# on demand, never assumed to exist.
export HOME=/lfs/local/0/$USER
mkdir -p "$HOME" || die "no node-local scratch at $HOME on $(hostname -s)"
export TMPDIR=/tmp/$USER
export XDG_CACHE_HOME=$HOME/.cache
export WANDB_DIR=$HOME/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# One pixi install for the whole cluster, on shared storage. Project
# environments stay node-local: detached-environments = false in
# $PIXI_HOME/config.toml keeps each env inside its project dir, and the project
# is a node-local clone under $TMPDIR. The package cache is node-local too, and
# warm across jobs landing on the same node.
export PIXI_HOME=$SHARED/.pixi
export PIXI_CACHE_DIR=$HOME/.cache/rattler/cache
export PATH=$PIXI_HOME/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # points into a home that is not this job's home
command -v pixi >/dev/null || die "pixi not found under $PIXI_HOME"

# Tokens come from the shared secrets dir rather than the job env, where slurm
# would record them.
_secrets=$SHARED/.secrets
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

echo "slurm-env: host=$(hostname -s) HOME=$HOME PIXI_HOME=$PIXI_HOME"
