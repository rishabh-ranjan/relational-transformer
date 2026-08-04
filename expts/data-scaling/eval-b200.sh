#!/bin/bash
# Evaluate one data-scaling model (its best_clf and best_reg checkpoints,
# sequentially) on 1xB200 under the il-lo QOS, following the
# expts/rebuttal/eval-node.sh pattern: submit from a clean, pushed checkout;
# the job clones the recorded commit fresh into /tmp and runs
# expts/data-scaling/eval.py under torchrun. The base config lives in that
# eval.py; this script only sets the checkpoint, the run id, and the
# per-checkpoint-kind task list.
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

    source expts/slurm-lib.sh
    rt_preflight
    echo "model:  $RT_MODEL"
    rt_submit "$LOG_DIR" \
        --job-name="rt-ds-eval-$RT_MODEL" \
        --nodelist=blackwell1 \
        --gres=gpu:b200:1 --cpus-per-task=36 --mem=375000M \
        --output="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_${RT_MODEL}_%j.out" \
        -- "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID model=$RT_MODEL"

# The clone is the one thing that cannot come from the shared library: the
# library lives in the repo, which does not exist on this node yet.
export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
mkdir -p "$TMPDIR/clones"
WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-ds-eval-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-lib.sh
rt_build_env "$LOG_DIR"

# Each checkpoint is evaluated only on tasks of its own kind, via the split
# task lists (forecast.json partitioned by task type).
declare -A TASK_LIST=(
    [best_clf]=expts/rebuttal/forecast-clf.json
    [best_reg]=expts/rebuttal/forecast-reg.json
)
for ckpt in best_clf best_reg; do
    id="${RT_RUN_ID}-${RT_MODEL}-${ckpt}"
    start=$(date +%s)
    echo "TIMING start model=$RT_MODEL ckpt=$ckpt id=$id epoch=$start ($(date -Is))"
    # --standalone picks a free rendezvous port, so concurrent single-GPU
    # eval jobs on the same node cannot collide.
    pixi run torchrun \
        --standalone --nnodes=1 --nproc-per-node=1 \
        expts/data-scaling/eval.py \
        --model.load-ckpt-path "$CKPT_ROOT/$RT_MODEL/$ckpt.safetensors" \
        --logger.id "$id" \
        --eval.db-task-list "${TASK_LIST[$ckpt]}" \
        "$@"
    end=$(date +%s)
    echo "TIMING end   model=$RT_MODEL ckpt=$ckpt id=$id epoch=$end elapsed_s=$((end - start))"
done
echo "=== $(date -Is) done ==="
