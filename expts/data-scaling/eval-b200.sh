#!/bin/bash
# Evaluate one data-scaling model (its best_clf and best_reg checkpoints,
# sequentially) on 1xB200 under the il-lo QOS, following the
# expts/rebuttal/eval-node.sh pattern: submit from a clean, pushed checkout;
# the job clones the recorded commit fresh into /tmp and runs
# expts/rebuttal/eval.py under torchrun with the data-scaling training-eval
# hparams (ctx 4096, tokens_per_gpu 2**17, items_per_task 1024) on the test
# split.
#
#   RT_MODEL=10pct ./expts/data-scaling/eval-b200.sh
#   RT_MODEL=32pct ./expts/data-scaling/eval-b200.sh
#   RT_MODEL=rt-j  ./expts/data-scaling/eval-b200.sh
#
# Checkpoints are read from
# /dfs/user/ranjanr/share/relational-transformer/expts/data-scaling/$RT_MODEL/.
# Extra args go to eval.py.

#SBATCH --job-name=rt-data-scaling-eval
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --qos=il-lo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-data-scaling-eval
CKPT_ROOT=/dfs/user/ranjanr/share/relational-transformer/expts/data-scaling

# ---------------------------------------------------------------- submit side
if [[ -z "${RT_RUN_ID:-}" ]]; then
    : "${RT_MODEL:?set RT_MODEL=10pct, 32pct, or rt-j}"

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

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "model:  $RT_MODEL"
    echo "run id: $RT_RUN_ID"

    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --job-name="rt-ds-eval-$RT_MODEL" \
        --gres=gpu:b200:1 --cpus-per-task=36 --mem=375000M \
        --output="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID",RT_MODEL="$RT_MODEL" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID model=$RT_MODEL"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-ds-eval-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-env.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

RUN_LOCK=$LOG_DIR/${RT_RUN_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

for ckpt in best_clf best_reg; do
    id="${RT_RUN_ID}-${RT_MODEL}-${ckpt}"
    start=$(date +%s)
    echo "TIMING start model=$RT_MODEL ckpt=$ckpt id=$id epoch=$start ($(date -Is))"
    pixi run torchrun \
        --nnodes=1 --nproc-per-node=1 \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        expts/rebuttal/eval.py \
        --model.load-ckpt-path "$CKPT_ROOT/$RT_MODEL/$ckpt.safetensors" \
        --logger.id "$id" \
        --eval.splits test \
        --eval.ctx-size-list 4096 \
        --eval.tokens-per-gpu 131072 \
        --eval.items-per-task 1024 \
        "$@"
    end=$(date +%s)
    echo "TIMING end   model=$RT_MODEL ckpt=$ckpt id=$id epoch=$end elapsed_s=$((end - start))"
    MASTER_PORT=$((MASTER_PORT + 1))
done
echo "=== $(date -Is) done ==="
