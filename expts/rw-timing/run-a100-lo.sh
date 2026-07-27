#!/bin/bash
# Run expts/rw-timing/timing.py on 1 A100 under il-lo, following the
# expts/data-scaling job pattern: submit from a clean, pushed checkout; the
# submitter records repo URL + HEAD commit and resubmits itself with sbatch;
# the job clones that commit fresh into /tmp on the compute node, builds the
# sampler, and runs timing.py. Extra args go to timing.py.
#
#   ./expts/rw-timing/run-a100-lo.sh [extra timing.py args...]

#SBATCH --job-name=rt-rw-timing
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --qos=il-lo
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
# 14 = site cap of CPUs per GPU on ampere for non-exclusive jobs.
#SBATCH --cpus-per-task=14
# The submit dir is node-local to the submit node, so don't try to start in it.
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-rw-timing

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

    RT_RUN_ID=$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "run id: $RT_RUN_ID"

    # Only the RT_* vars travel; the job rebuilds its env below. Strip inherited
    # SLURM/SBATCH vars so a submit from inside an allocation doesn't impose
    # that job's shape on this one.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
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

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-rw-timing-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
# The -c url.insteadOf rewrite authenticates the fetch without writing the
# token into the clone's .git/config (the remote keeps the plain URL).
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

pixi run python expts/rw-timing/timing.py "$@"

echo "=== $(date -Is) done ==="
