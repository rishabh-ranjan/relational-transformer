#!/bin/bash
# Launch expts/data-scaling/train.py on 2 B200s of blackwell1, to compare
# training throughput against the 8xA100 (ampere) run. Same train config: the
# code adapts to the GPU count on its own and total_bs is held constant, so
# sec/step is comparable modulo the changed grad-accum.
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

#SBATCH --job-name=rt-data-scaling-b200
#SBATCH --partition=il
#SBATCH --account=infolab
# 2 GPUs, not more: the default `il` QOS caps b200 at 2 per user (and
# il-interactive caps GPUs at 2 of any kind), so 4 is only reachable under il-lo
# -- priority 100 instead of 1000, which on a busy blackwell1 means waiting for
# days. Two B200s under `il` keep the fast queue.
#SBATCH --qos=il
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=blackwell1
#SBATCH --gres=gpu:b200:2
#SBATCH --ntasks-per-node=1
# A quarter of the node's GPUs, so a quarter of its cores; not --exclusive.
#SBATCH --cpus-per-task=72
# Ask for memory explicitly: the site's job_submit plugin would default this to
# mem-per-gpu=144723M (~283G for 2 GPUs), well below the ~603G the mixture is
# populated into the page cache, so the cache would be evicted mid-run and reads
# would fall back to /dfs. 770000M is the most MaxMemPerCPU (10700M) allows at
# 72 CPUs.
#SBATCH --mem=770000M
# The submit dir is node-local to the submit node, so don't try to start in it.
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@300

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-data-scaling-b200

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

    # Run id == wandb id == output subdir; fixed here so requeues resume.
    RT_RUN_ID=$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')

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
    # Slurm env vars outrank #SBATCH directives, so a submit from inside an
    # interactive allocation would silently impose that job's cpu/mem/node
    # shape on this one. Strip them.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --output="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID"

# Bootstrap only what the clone itself needs; the full env comes from the
# cloned tree's expts/slurm-env.sh below, so the recorded commit pins the job
# environment too.
export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
# The pixi env lives inside the clone, so it goes away with it.
trap 'rm -rf "$WORK_DIR"' EXIT
# The -c url.insteadOf rewrite authenticates the fetch without writing the
# token into the clone's .git/config (the remote keeps the plain URL).
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

# One shared definition of the job environment (node-local HOME, PATH, caches,
# tokens); see expts/slurm-env.sh. Nothing env-related is set per job script.
source expts/slurm-env.sh

# Static rendezvous with a per-job port (dynamic c10d has wedged under load).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# pixi.lock is gitignored in this repo, so a fresh clone has none and the first
# job of a run has to solve the environment itself. Keep that solve next to the
# run's logs and reuse it on every requeue, so the environment is identical
# across the whole run instead of re-solving (and drifting) each time.
RUN_LOCK=$LOG_DIR/${RT_RUN_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

pixi run torchrun \
    --nnodes=1 --nproc-per-node=2 \
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
