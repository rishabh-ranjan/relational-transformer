#!/bin/bash
# Launch expts/data-scaling/train.py as a full-node DDP job on one ampere node.
#
#   ./expts/data-scaling/single-node.sh [extra train.py args...]
#
# Run it from a clean, pushed checkout: the submitter records the repo URL and
# HEAD commit, refuses to submit a dirty or unpushed tree, mints a run id with
# rt.config.timestamp(), and submits itself with sbatch. The job clones the repo
# fresh into a unique /tmp dir, checks out that commit, and runs train.py under
# torchrun with --logger.id=<run id>. The id is fixed at submit time, so every
# requeue (preemption or time limit) reuses the same wandb run and output dir
# and resumes from resume.pt there.

#SBATCH --job-name=rt-data-scaling
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=ampere
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --propagate=MEMLOCK
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@300

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-data-scaling

# ---------------------------------------------------------------- submit side
# RT_RUN_ID (not SLURM_JOB_ID) tells the two sides apart: submitting from inside
# an interactive allocation means SLURM_JOB_ID is already set in your shell.
if [[ -z "${RT_RUN_ID:-}" ]]; then
    cd "$(git rev-parse --show-toplevel)"

    if [[ -n "$(git status --porcelain)" ]]; then
        echo "WARNING: working tree is dirty; commit or stash before submitting." >&2
        git status --short >&2
        exit 1
    fi

    RT_REPO=$(git remote get-url origin)
    RT_COMMIT=$(git rev-parse HEAD)
    RT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    git fetch --quiet origin
    if ! git merge-base --is-ancestor "$RT_COMMIT" "origin/$RT_BRANCH" 2>/dev/null; then
        echo "WARNING: $RT_COMMIT is not on origin/$RT_BRANCH; push before submitting." >&2
        exit 1
    fi

    # Run id == wandb id == output subdir; frozen here so requeues resume.
    RT_RUN_ID=$(pixi run --frozen python -c 'from rt.config import timestamp; print(timestamp())')

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "run id: $RT_RUN_ID"

    # Deliberately NOT --export=ALL: this shell's env is fish-config'd for the
    # submit node (HOME=/lfs/local/0/..., a node-local PATH) and holds API
    # tokens. Exporting it would point the job at a home that does not exist on
    # the compute node and would stash the tokens in Slurm's job record. Only
    # the RT_* vars travel; the job rebuilds its env below.
    exec sbatch \
        --output="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID"

# ---- env: the batch script is not a login shell, so no fish config runs ----
# Node-local home (pixi envs, HF/torch caches) on whichever node we landed on.
export USER=${USER:-$(id -un)}
NODE_HOME=/lfs/local/0/$USER
mkdir -p "$NODE_HOME" || NODE_HOME=/sailhome/$USER
export HOME=$NODE_HOME
export PATH=$HOME/.pixi/bin:/sailhome/$USER/.pixi/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # submit-node-local, would shadow the pixi env
export TMPDIR=/tmp/$USER
export XDG_CACHE_HOME=$HOME/.cache
export WANDB_DIR=$HOME/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HOME/.pixi"

# ---- tokens: read from the shared secrets dir, never from the job env ----
SECRETS=/sailhome/$USER/.secrets
read_secret() { tr -d '[:space:]' < "$SECRETS/$1"; }
WANDB_API_KEY=$(read_secret wandb); export WANDB_API_KEY
HF_TOKEN=$(read_secret huggingface); export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
# github: only needed if the repo ever goes private (the clone below is https).
GITHUB_TOKEN=$(read_secret github); export GITHUB_TOKEN GH_TOKEN=$GITHUB_TOKEN
echo "env: HOME=$HOME pixi=$(command -v pixi) tokens=wandb,hf,github"

WORK_DIR=$(mktemp -d "/tmp/ranjanr/clones/rt-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
# The -c url.insteadOf rewrite authenticates the fetch without writing the
# token into the clone's .git/config (the remote keeps the plain URL).
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

ulimit -l unlimited || true
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# Static rendezvous with a per-job port (dynamic c10d has wedged under load).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

pixi run --frozen build-sampler

pixi run --frozen torchrun \
    --nnodes=1 --nproc-per-node=8 \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    expts/data-scaling/train.py \
    --logger.id "$RT_RUN_ID" \
    "$@" &
TRAIN_PID=$!

# Slurm sends SIGTERM on preemption (GraceTime=300s) and SIGUSR1 300s before the
# time limit. Forward either to torchrun, which shuts the workers down; the
# train loop then writes resume.pt and exits 0. Preemption requeues the job by
# itself (PreemptMode=REQUEUE); the time-limit case we requeue explicitly.
requeue_after_wait() {
    local sig=$1
    echo "=== $(date -Is) caught $sig; forwarding to torchrun ==="
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    if [[ $sig == SIGUSR1 ]]; then
        echo "=== requeueing job $SLURM_JOB_ID (time limit) ==="
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit 0
}
trap 'requeue_after_wait SIGTERM' TERM
trap 'requeue_after_wait SIGUSR1' USR1

wait "$TRAIN_PID"
