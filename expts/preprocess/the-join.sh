#!/bin/bash
# Preprocess all of stanford-star/the-join into rustler's on-disk format, as a
# sharded slurm job array.
#
#   ./expts/preprocess/the-join.sh [extra `preprocess many` args...]
#
# Run it from a clean, pushed checkout: the submitter records the repo URL and
# HEAD commit, refuses to submit a dirty or unpushed tree, and submits itself
# with sbatch. Each array task clones the repo fresh into a unique /tmp dir,
# checks out that commit, and runs `rt.cli.preprocess many` on its shard.
#
# Shape of the work: ~650 databases, each independent -- download from the Hub,
# rustler `pre` (rayon-multithreaded, memory-hungry on the big dbs), then text
# embeddings on the GPU. So one task per shard on a *fraction* of a node (one
# GPU, 32 CPUs, 320G) beats whole-node jobs: MiniLM saturates nothing, and many
# small tasks queue in the gaps between the big exclusive training jobs.
#
# Resumability is `--skip-existing`, which skips a database whose embeddings are
# already written *and recorded in its meta.json* -- a database interrupted
# mid-preprocess is redone rather than left half-written. That makes the job
# array requeue-safe (preemption on `il` is REQUEUE), makes a rerun of this
# script a cheap way to mop up the failures of an earlier pass, and means shards
# need not be balanced: an empty shard exits in seconds.
#
# OUT_DIR is a fresh directory on purpose. /dfs/user/$USER/pre/the-join-preprocessed
# is the downloaded copy of the published stanford-star/the-join-preprocessed and
# is never written to here.

#SBATCH --job-name=rt-pre-the-join
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# 320G/32cpu = 10240M per cpu, just under the partition's MaxMemPerCPU (10700M);
# asking for more memory per cpu makes slurm silently widen the cpu request.
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --chdir=/tmp
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-pre-the-join
OUT_DIR=${RT_OUT_DIR:-/dfs/user/ranjanr/pre/the-join-preprocessed-v2}
REPO_ID=${RT_HF_REPO:-stanford-star/the-join}
NUM_SHARDS=${RT_NUM_SHARDS:-32}
# %8: at most 8 tasks in flight, so this pass leaves room for training jobs.
ARRAY=${RT_ARRAY:-0-$((NUM_SHARDS - 1))%8}

# ---------------------------------------------------------------- submit side
if [[ -z "${RT_COMMIT:-}" ]]; then
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

    mkdir -p "$LOG_DIR" "$OUT_DIR"

    # The task lists ship with the preprocessed data (see rt.data.tasks), so
    # carry them over from the published collection once, here, rather than
    # racing 8 array tasks to copy the same directory. They name (db, task)
    # pairs, which a fresh preprocess of the same source reproduces verbatim.
    SRC_LISTS=/dfs/user/$USER/pre/the-join-preprocessed/db-task-lists
    if [[ -d $SRC_LISTS && ! -d $OUT_DIR/db-task-lists ]]; then
        cp -r "$SRC_LISTS" "$OUT_DIR/db-task-lists"
        echo "copied db-task-lists from $SRC_LISTS"
    fi

    echo "repo:    $RT_REPO"
    echo "branch:  $RT_BRANCH"
    echo "commit:  $RT_COMMIT"
    echo "hf repo: $REPO_ID"
    echo "out dir: $OUT_DIR"
    echo "array:   $ARRAY ($NUM_SHARDS shards)"

    # Deliberately NOT --export=ALL: this shell's env is fish-config'd for the
    # submit node (a node-local HOME and PATH) and holds API tokens. Only the
    # RT_* vars travel; the job rebuilds its env from the cloned tree.
    # Slurm env vars outrank #SBATCH directives, so a submit from inside an
    # interactive allocation would impose that job's shape on this one. Strip.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --array="$ARRAY" \
        --output="$LOG_DIR/%A_%a.out" \
        --error="$LOG_DIR/%A_%a.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_OUT_DIR="$OUT_DIR",RT_HF_REPO="$REPO_ID",RT_NUM_SHARDS="$NUM_SHARDS" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
SHARD=${SLURM_ARRAY_TASK_ID:-0}
echo "=== $(date -Is) job $SLURM_JOB_ID shard $SHARD/$NUM_SHARDS on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT out_dir=$OUT_DIR"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-pre-job${SLURM_JOB_ID}.XXXX")
# The pixi env lives inside the clone, so it goes away with it.
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

# One shared definition of the job environment (node-local HOME, PATH, caches,
# tokens); see expts/slurm-env.sh. The raw Hub downloads land in that node-local
# cache, so shards never contend on /dfs for their inputs -- only the
# preprocessed output is written there, because training reads it from every node.
source expts/slurm-env.sh

# pixi.lock is gitignored, so a fresh clone has none and would re-solve the
# environment in every one of the array's tasks. Solve once, keep the lock next
# to the logs, and reuse it -- identical environments across the whole pass.
RUN_LOCK=$LOG_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

mkdir -p "$OUT_DIR"
pixi run python -m rt.cli.preprocess many \
    --repo "$REPO_ID" \
    --out-dir "$OUT_DIR" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --skip-existing \
    "$@"

echo "=== $(date -Is) shard $SHARD done ==="
