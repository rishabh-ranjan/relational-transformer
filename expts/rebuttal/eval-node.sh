#!/bin/bash
# DDP-evaluate one pretrained RT model (both its classification and regression
# checkpoints, sequentially) on a full node, following the
# expts/data-scaling/single-node*.sh pattern: submit from a clean, pushed
# checkout; the job clones the recorded commit fresh into /tmp and runs
# expts/rebuttal/eval.py under torchrun.
#
#   RT_MODEL=rt-j RT_NODE=ampere8    ./expts/rebuttal/eval-node.sh [eval.py args...]
#   RT_MODEL=rt-p RT_NODE=blackwell1 ./expts/rebuttal/eval-node.sh [eval.py args...]
#
# Extra args go to eval.py, e.g. a smaller task list for a timing test run:
#   --eval.db-task-list expts/rebuttal/rel-f1.json
# (relative paths are resolved in the job's fresh clone).
#
# Timings: the job brackets each checkpoint eval with `TIMING ...` lines
# (epoch seconds + elapsed), for projecting the full-run ETA from a test run.
#
# Eval has no resume state, so unlike the training scripts there is no
# checkpoint-on-preemption machinery; --requeue simply reruns from scratch
# (rank 0 rewrites the submission CSVs, so a rerun is idempotent).

#SBATCH --job-name=rt-rebuttal-eval
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# The submit dir is node-local to the submit node, so don't try to start in it.
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-rebuttal-eval

# ---------------------------------------------------------------- submit side
if [[ -z "${RT_RUN_ID:-}" ]]; then
    : "${RT_MODEL:?set RT_MODEL=rt-j or rt-p}"
    : "${RT_NODE:?set RT_NODE=ampere8 or blackwell1}"

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

    RT_RUN_ID=$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')

    # Per-node shape, both under the default `il` QOS. ampere: 8xA100 (within
    # the 10-a100/user cap); 112 = non-exclusive CPU cap (8 x 14). blackwell:
    # `il` caps b200 at 2/user (QOSMaxGRESPerUser; 8xB200 needs il-lo), and
    # memory must be asked for explicitly or the site plugin under-defaults it.
    case "$RT_NODE" in
        ampere*)
            shape=(--gres=gpu:a100:8 --cpus-per-task=112) ;;
        blackwell*)
            shape=(--gres=gpu:b200:2 --cpus-per-task=72 --mem=750000M) ;;
        *)
            echo "unknown RT_NODE=$RT_NODE" >&2; exit 1 ;;
    esac

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "model:  $RT_MODEL  node: $RT_NODE"
    echo "run id: $RT_RUN_ID"

    # Same env hygiene as the data-scaling submitters: don't --export=ALL, and
    # strip inherited SLURM_/SBATCH_ vars so a submit from inside an interactive
    # allocation doesn't impose that job's shape.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --job-name="rt-rebuttal-eval-$RT_MODEL" \
        --nodelist="$RT_NODE" \
        "${shape[@]}" \
        --output="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID",RT_MODEL="$RT_MODEL",RT_CKPTS="${RT_CKPTS:-}" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID model=$RT_MODEL"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-eval-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-env.sh

export MASTER_ADDR=127.0.0.1
# A free port per launch, asked from the kernel. The job-id arithmetic the
# training scripts use collided here: jobs N and N+1 on one node clash as soon
# as job N's second phase computes port(N)+1 == port(N+1) (jobs 99539/99540).
free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'
}

RUN_LOCK=$LOG_DIR/${RT_RUN_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

NPROC=${SLURM_GPUS_ON_NODE:-8}
# Each checkpoint is evaluated only on tasks of its own kind, via the split
# task lists (forecast.json partitioned by task type) -- unless the caller
# passed an explicit --eval.db-task-list, which then applies to both phases.
declare -A TASK_LIST=(
    [classification]=expts/rebuttal/forecast-clf.json
    [regression]=expts/rebuttal/forecast-reg.json
)
# RT_CKPTS overrides which checkpoints run (e.g. RT_CKPTS=regression to redo
# just one phase).
for task_type in ${RT_CKPTS:-classification regression}; do
    id="${RT_RUN_ID}-${RT_MODEL}-${task_type}"
    list_arg=(--eval.db-task-list "${TASK_LIST[$task_type]}")
    case " $* " in *" --eval.db-task-list"*) list_arg=() ;; esac
    MASTER_PORT=$(free_port)
    export MASTER_PORT
    start=$(date +%s)
    echo "TIMING start model=$RT_MODEL ckpt=$task_type id=$id epoch=$start ($(date -Is))"
    pixi run torchrun \
        --nnodes=1 --nproc-per-node="$NPROC" \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        expts/rebuttal/eval.py \
        --model.load-ckpt-path "stanford-star/$RT_MODEL/$task_type" \
        --logger.id "$id" \
        "${list_arg[@]}" \
        "$@"
    end=$(date +%s)
    echo "TIMING end   model=$RT_MODEL ckpt=$task_type id=$id epoch=$end elapsed_s=$((end - start))"
done
echo "=== $(date -Is) done ==="
