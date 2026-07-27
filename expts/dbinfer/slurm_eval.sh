#!/bin/bash
# Stage 3: the context-scaling eval -- one full-node DDP job per method.
#
#   RT_METHOD=rt ./expts/dbinfer/slurm_eval.sh                     # submit one method
#   RT_METHOD=rt RT_QOS=il ./expts/dbinfer/slurm_eval.sh            # on the priority QOS
#   RT_METHOD=rdblearn_tabicl RT_DEPEND=afterok:12345 ./...sh       # chained behind a stage
#
# Normally driven by ``launch.sh``, which submits every stage at once with the
# right dependencies. Same submit contract as the other stages: clean pushed tree,
# recorded commit, fresh clone per job.
#
# One job per *method*, not per (method, task): `eval.py` loops the task list
# internally and builds each target's context once at ctx=8192, reading all six
# context points off as prefixes. So a single job produces a whole table row, and
# there are no per-ctx output directories and no column-merge step -- which is what
# the RelBench campaign's launcher needed one job per (task, ctx) to work around.
#
# 8 a100s on one ampere node, under torchrun: `eval.py` shards items across ranks
# and gathers on rank 0. ampere4 is excluded (it hosts the data-scaling training
# run). The `il` QOS caps a user at gres/gpu:a100=10, i.e. exactly one such node,
# so the second method goes on `il-lo` -- see RT_QOS.
#
# Resumability: eval.py skips a (method, task) whose JSON exists, so a requeue after
# preemption resumes at the next task rather than redoing the row.

#SBATCH --job-name=dbinfer-eval
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=ampere
#SBATCH --exclude=ampere4
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
# No --mem: the partition caps at MaxMemPerCPU (10700M x 128), so --mem=0 is
# rejected outright, while --exclusive plus DefMemPerGPU=240000 lands on the whole
# node anyway and nothing else can run there.
#SBATCH --chdir=/tmp
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/$USER/slurm-logs/dbinfer-eval
OUT_DIR=${RT_OUT_DIR:-/dfs/user/$USER/dbinfer-scaling}
PRE_DIR=${RT_PRE_DIR:-/dfs/user/$USER/pre/dbinfer-preprocessed}
METHOD=${RT_METHOD:?set RT_METHOD=rt|rdblearn_tabicl}
NPROC=${RT_NPROC:-8}

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

    EXTRA=()
    [[ -n ${RT_QOS:-} ]] && EXTRA+=(--qos="$RT_QOS")
    [[ -n ${RT_DEPEND:-} ]] && EXTRA+=(--dependency="$RT_DEPEND")

    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --job-name="dbinfer-eval-$METHOD" \
        --output="$LOG_DIR/%j-$METHOD.out" \
        --error="$LOG_DIR/%j-$METHOD.out" \
        "${EXTRA[@]}" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_OUT_DIR="$OUT_DIR",RT_PRE_DIR="$PRE_DIR",RT_METHOD="$METHOD",RT_NPROC="$NPROC" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID method=$METHOD on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT pre_dir=$PRE_DIR out_dir=$OUT_DIR nproc=$NPROC"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/dbinfer-eval-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-env.sh

RUN_LOCK=$LOG_DIR/eval.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

# ctx=8192 with materialized attention masks is O(ctx^2) memory; the flex-attention
# path is the one that fits, same as the RelBench long-context runs.
export RT_MATERIALIZE_ATTN_MASKS=0

pixi run torchrun --standalone --nproc-per-node="$NPROC" \
    expts/dbinfer/eval.py \
    --method "$METHOD" \
    --pre-dir "$PRE_DIR" \
    --out-dir "$OUT_DIR" \
    "$@"

echo "=== $(date -Is) $METHOD done ==="
