#!/bin/bash
# Stage 1: preprocess the four 4DBInfer databases into rustler's on-disk format.
#
#   ./expts/dbinfer/slurm_preprocess.sh [extra `preprocess one` args...]
#
# Same contract as expts/preprocess/the-join.sh: run it from a clean, pushed
# checkout -- the submitter records the repo URL and HEAD commit, refuses a dirty
# or unpushed tree, and submits itself with sbatch. Each array task clones the
# repo fresh into a unique /tmp dir, checks out that commit, and preprocesses one
# database: rustler `pre` (rayon-multithreaded, memory-hungry) then MiniLM text
# embeddings on the GPU.
#
# One task per database rather than per shard: there are only four, and they are
# very unevenly sized (amazon is 5.3 GiB and holds a 13.7M-row Review table with
# text; diginetica's QueryResult is 92M rows), so sharding by count would just
# put the two big ones in one shard anyway.
#
# Resumability is `--skip-existing`, which skips a database whose embeddings are
# written *and* recorded in its meta.json -- a database interrupted mid-preprocess
# is redone rather than left half-written. That makes the array requeue-safe
# (preemption on `il` is REQUEUE) and a rerun a cheap way to mop up failures.

#SBATCH --job-name=dbinfer-pre
#SBATCH --partition=il
# il-lo, not il: the `il` QOS caps a user at gres/gpu:a100=10 (and gres/gpu=10),
# which is exactly one 8-GPU ampere node -- reserved for the eval. Preprocessing
# needs a GPU only for the MiniLM text pass, so it takes the low-priority QOS and
# stays off the ampere nodes entirely.
#SBATCH --qos=il-lo
#SBATCH --exclude=ampere[1-9],blackwell1,hyperion[1,3]
# 48G picks the rtx8000 nodes over the 12G 2080tis. The MiniLM pass is the memory
# peak, and amazon's 21.2M texts include long review bodies: at batch_size 1024 it
# OOMed on a 10.58 GiB 2080ti. Excluding the amperes (so the eval owns them) leaves
# only these two GPU sizes, so the larger one has to be asked for explicitly.
#SBATCH --constraint=48G
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

LOG_DIR=/dfs/user/$USER/slurm-logs/dbinfer-pre
OUT_DIR=${RT_OUT_DIR:-/dfs/user/$USER/pre/dbinfer-preprocessed}
# The corrected dbinfer collection (see provenance/dbinfer.py in the relbench repo).
REPO_ID=${RT_HF_REPO:-stanford-star/dbinfer}
DBS=(dbinfer-amazon dbinfer-diginetica dbinfer-retailrocket dbinfer-stackexchange)
# RT_ARRAY narrows the array to a subset of DBS by index, for redoing one database
# without touching the others (e.g. RT_ARRAY=0 for amazon alone).
ARRAY=${RT_ARRAY:-0-$((${#DBS[@]} - 1))%4}

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
    echo "repo:    $RT_REPO"
    echo "commit:  $RT_COMMIT"
    echo "hf repo: $REPO_ID"
    echo "out dir: $OUT_DIR"
    echo "dbs:     ${DBS[*]}"
    echo "array:   $ARRAY"

    # Deliberately NOT --export=ALL: this shell's env is fish-config'd for the
    # submit node (node-local HOME and PATH) and holds API tokens. Only the RT_*
    # vars travel; the job rebuilds its env from the cloned tree. Slurm env vars
    # outrank #SBATCH directives, so a submit from inside an interactive
    # allocation would impose that job's shape on this one -- strip them.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    EXTRA=()
    [[ -n ${RT_DEPEND:-} ]] && EXTRA+=(--dependency="$RT_DEPEND")

    exec env "${strip[@]}" sbatch \
        --array="$ARRAY" \
        "${EXTRA[@]}" \
        --output="$LOG_DIR/%A_%a.out" \
        --error="$LOG_DIR/%A_%a.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_OUT_DIR="$OUT_DIR",RT_HF_REPO="$REPO_ID" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
DB=${DBS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== $(date -Is) job $SLURM_JOB_ID db=$DB on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT out_dir=$OUT_DIR"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/dbinfer-pre-job${SLURM_JOB_ID}.XXXX")
# The pixi env lives inside the clone, so it goes away with it.
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

# One shared definition of the job environment (node-local HOME, PATH, caches,
# tokens). Hub downloads land in the node-local cache, so tasks never contend on
# /dfs for inputs -- only the preprocessed output is written there, because eval
# reads it from every node.
source expts/slurm-env.sh

# pixi.lock is gitignored, so a fresh clone would re-solve the environment in
# every array task. Solve once, keep the lock next to the logs, reuse it.
RUN_LOCK=$LOG_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

# Stage the dataset node-locally and keep only the tasks this experiment uses.
#
# rustler's preprocess computes each table's column stats once and applies them
# *positionally*, so a task whose splits carry different columns misaligns them.
# 4DBInfer's retrieval tasks do exactly that: `purchase/train` holds positives only
# while val/test add `label` and `query_idx`, so position 2 is a datetime in train
# (mean=0, std=0) and int64 `label` in val -- normalizing gives (label-0)/0 = inf
# and pre.rs panics on it. The numeric arm's zero-std guard cannot help, because the
# stat came from the datetime arm.
#
# Those are link-prediction tasks, excluded from this experiment anyway, so drop
# their directories instead of preprocessing them. Deriving the keep-list from
# tasks.json means this needs no per-db special-casing.
DS_ROOT=$TMPDIR/dbinfer-ds-job${SLURM_JOB_ID}
mkdir -p "$DS_ROOT"
pixi run hf download "$REPO_ID" --repo-type dataset \
    --include "$DB/*" --local-dir "$DS_ROOT" >/dev/null
DS_DIR=$DS_ROOT/$DB

KEEP=$(python3 -c "
import json, sys
pairs = json.load(open('expts/dbinfer/tasks.json'))
print('\n'.join(t for db, t in pairs if db == '$DB'))")
echo "tasks kept for $DB: $(tr '\n' ' ' <<<"$KEEP")"
for d in "$DS_DIR"/tasks/*/; do
    [[ -d $d ]] || continue
    name=$(basename "$d")
    if ! grep -qx "$name" <<<"$KEEP"; then
        echo "dropping task dir not used by this experiment: $name"
        rm -rf "$d"
    fi
done

# Smaller embedding batches than the 1024 default: the peak is text length, not
# row count, and amazon's reviews are long. 256 fits comfortably on a 48G card and
# costs little -- MiniLM saturates neither GPU at this size.
pixi run python -m rt.cli.preprocess one \
    --dataset "$DS_DIR" \
    --out-dir "$OUT_DIR" \
    --batch-size "${RT_EMBED_BATCH:-256}" \
    "$@"

echo "=== $(date -Is) $DB done ==="
