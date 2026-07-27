#!/bin/bash
# Stage 3: the context-scaling eval -- one array task per (method, task).
#
#   ./expts/dbinfer/slurm_eval.sh                        # all 12 points
#   RT_METHODS=rt ./expts/dbinfer/slurm_eval.sh          # rt-j only (the priority)
#   RT_TASKS=dbinfer-amazon/churn ./expts/dbinfer/slurm_eval.sh
#
# Same submit contract as the other two stages. The array is (methods x tasks),
# **not** (methods x tasks x ctx): `eval.py` builds each target's context once at
# ctx=8192 and reads all six context points off as prefixes, so one job produces a
# task's whole row. That is what removes the per-ctx output directories and the
# merge step the RelBench campaign's launcher needed.
#
# Resumability: eval.py skips a (method, task) whose JSON exists, so a requeue or a
# rerun tops up what is missing. Jobs already queued for a point are not
# re-submitted (see the squeue filter below), so a top-up does not burn a GPU
# recomputing something in flight.

#SBATCH --job-name=dbinfer-eval
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --chdir=/tmp
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/$USER/slurm-logs/dbinfer-eval
OUT_DIR=${RT_OUT_DIR:-/dfs/user/$USER/dbinfer-scaling}
PRE_DIR=${RT_PRE_DIR:-/dfs/user/$USER/pre/dbinfer-preprocessed}
METHODS=${RT_METHODS:-"rt rdblearn_tabicl"}

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

    TASKS=${RT_TASKS:-$(python3 -c '
import json, sys
for db, t in json.load(open("expts/dbinfer/tasks.json")):
    print(f"{db}/{t}")' | tr "\n" " ")}

    # Build the (method, task) point list, dropping points already done or queued.
    INFLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep "^dbinfer-eval-" || true)
    POINTS=()
    for m in $METHODS; do
        for t in $TASKS; do
            json="$OUT_DIR/$m/${t/\//__}.json"
            if [[ -f $json ]]; then
                echo "skip (done):     $m $t"
                continue
            fi
            name="dbinfer-eval-$m-${t//\//_}"
            if grep -qx "$name" <<<"$INFLIGHT"; then
                echo "skip (in queue): $m $t"
                continue
            fi
            POINTS+=("$m|$t")
        done
    done
    if [[ ${#POINTS[@]} -eq 0 ]]; then
        echo "nothing to submit."
        exit 0
    fi

    mkdir -p "$LOG_DIR" "$OUT_DIR"
    printf 'point: %s\n' "${POINTS[@]}"
    echo "commit:  $RT_COMMIT"
    echo "out dir: $OUT_DIR"
    echo "points:  ${#POINTS[@]}"

    # One sbatch per point, so each carries its own job name -- that is what makes
    # the in-flight filter above work on a top-up. An array would share one name.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')
    for p in "${POINTS[@]}"; do
        m=${p%%|*}; t=${p##*|}
        env "${strip[@]}" sbatch \
            --job-name="dbinfer-eval-$m-${t//\//_}" \
            --output="$LOG_DIR/%j-$m-${t//\//_}.out" \
            --error="$LOG_DIR/%j-$m-${t//\//_}.out" \
            --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_OUT_DIR="$OUT_DIR",RT_PRE_DIR="$PRE_DIR",RT_METHOD="$m",RT_TASK="$t" \
            "$0" "$@"
    done
    exit 0
fi

# ------------------------------------------------------------------- job side
: "${RT_METHOD:?}"; : "${RT_TASK:?}"
echo "=== $(date -Is) job $SLURM_JOB_ID method=$RT_METHOD task=$RT_TASK on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="

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

pixi run python expts/dbinfer/eval.py \
    --method "$RT_METHOD" \
    --tasks "$RT_TASK" \
    --pre-dir "$PRE_DIR" \
    --out-dir "$OUT_DIR" \
    "$@"

echo "=== $(date -Is) $RT_METHOD $RT_TASK done ==="
