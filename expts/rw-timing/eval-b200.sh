#!/bin/bash
# Full test-set eval of RT-J on the 3 rel-f1 forecast tasks under two walk
# configs, (num_walks, walk_length) = (10000, 20) and (1000, 10), on 1xB200
# under il-lo. Follows expts/data-scaling/eval-b200.sh: submit from a clean,
# pushed checkout; the job clones the recorded commit fresh into /tmp and runs
# rt.cli.eval per (checkpoint kind, walk config). Metrics land in the slurm
# log; submission CSVs under ~/ckpts on the compute node.
#
#   ./expts/rw-timing/eval-b200.sh [extra rt.cli.eval args...]

#SBATCH --job-name=rt-rw-timing-eval
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --qos=il-lo
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-rw-timing
PRE_DIR=/dfs/user/ranjanr/pre/relbench-preprocessed

# ---------------------------------------------------------------- submit side
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

    RT_RUN_ID=$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "run id: $RT_RUN_ID"

    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --nodelist=blackwell1 \
        --gres=gpu:b200:1 --cpus-per-task=36 --mem=375000M \
        --output="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID"

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-rw-eval-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-env.sh

RUN_LOCK=$LOG_DIR/${RT_RUN_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

declare -A CKPT=(
    [clf]=stanford-star/rt-j/classification
    [reg]=stanford-star/rt-j/regression
)
declare -A TASK_LIST=(
    [clf]=expts/rw-timing/rel-f1-clf.json
    [reg]=expts/rw-timing/rel-f1-reg.json
)
for walks_len in "10000 20" "1000 10"; do
    read -r walks len <<< "$walks_len"
    for kind in clf reg; do
        id="${RT_RUN_ID}-${kind}-w${walks}-l${len}"
        start=$(date +%s)
        echo "TIMING start kind=$kind num_walks=$walks walk_length=$len id=$id epoch=$start ($(date -Is))"
        pixi run torchrun \
            --standalone --nnodes=1 --nproc-per-node=1 \
            -m rt.cli.eval \
            --model.load-ckpt-path "${CKPT[$kind]}" \
            --logger.id "$id" \
            --eval.pre-dir "$PRE_DIR" \
            --eval.db-task-list "${TASK_LIST[$kind]}" \
            --eval.num-walks "$walks" \
            --eval.walk-length "$len" \
            "$@"
        end=$(date +%s)
        echo "TIMING end   kind=$kind num_walks=$walks walk_length=$len id=$id epoch=$end elapsed_s=$((end - start))"
    done
done
echo "=== $(date -Is) done ==="
