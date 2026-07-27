#!/bin/bash
# Stage 2: write depth-2 DFS feature matrices for the four 4DBInfer databases.
#
#   ./expts/dbinfer/slurm_featurize.sh [extra featurize.py args...]
#
# Same submit contract as slurm_preprocess.sh (clean pushed tree, recorded commit,
# fresh clone per array task), with one difference that forces a separate script:
# **this stage does not run in the repo's pixi env.** rdblearn pins relbench==1.1.0
# and pulls autogluon, which conflict with this repo's relbench-hf, so the stage
# gets its own uv venv -- cached on shared storage and reused across array tasks.
#
# CPU-only: fastdfs's dfs2sql engine is duckdb, not a GPU workload.
#
# DBB_DATASET_HOME points at the raw 4DBInfer archives. dbinfer_bench downloads
# them there on first use (~7 GiB over the four databases, from data.dgl.ai) and
# reuses them after, so this lives on /dfs rather than node-local scratch.
#
# Resumability: featurize.py skips a table whose <table>_meta.json already exists,
# so a requeue or a rerun tops up whatever is missing.

#SBATCH --job-name=dbinfer-feat
#SBATCH --partition=il-cpu
#SBATCH --qos=il-cpu-long
#SBATCH --account=infolab
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --chdir=/tmp
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

LOG_DIR=/dfs/user/$USER/slurm-logs/dbinfer-feat
PRE_DIR=${RT_PRE_DIR:-/dfs/user/$USER/pre/dbinfer-preprocessed}
BUILD_DIR=${RT_BUILD_DIR:-}   # optional: published parquets, enables --verify-rows
VENV=${RT_VENV:-/dfs/user/$USER/venvs/rdblearn}
export DBB_DATASET_HOME=${DBB_DATASET_HOME:-/dfs/user/$USER/dbinfer-raw}
DBS=(dbinfer-amazon dbinfer-diginetica dbinfer-retailrocket dbinfer-stackexchange)

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

    if [[ ! -d $PRE_DIR ]]; then
        echo "WARNING: $PRE_DIR does not exist; run slurm_preprocess.sh first." >&2
        exit 1
    fi

    mkdir -p "$LOG_DIR" "$DBB_DATASET_HOME"
    echo "commit:   $RT_COMMIT"
    echo "pre dir:  $PRE_DIR"
    echo "archives: $DBB_DATASET_HOME"
    echo "venv:     $VENV"
    echo "dbs:      ${DBS[*]}"

    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --array="0-$((${#DBS[@]} - 1))%4" \
        --output="$LOG_DIR/%A_%a.out" \
        --error="$LOG_DIR/%A_%a.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_PRE_DIR="$PRE_DIR",RT_BUILD_DIR="$BUILD_DIR",RT_VENV="$VENV",DBB_DATASET_HOME="$DBB_DATASET_HOME" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
DB=${DBS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "=== $(date -Is) job $SLURM_JOB_ID db=$DB on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="

export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/dbinfer-feat-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-env.sh

# The rdblearn env, built once and shared. Serialize creation across array tasks:
# four of them starting at the same second would otherwise race on the same dir.
mkdir -p "$(dirname "$VENV")"
exec 9>"$(dirname "$VENV")/.rdblearn-venv.lock"; flock 9
if [[ ! -x $VENV/bin/python ]]; then
    echo "creating $VENV"
    uv venv --python 3.11 "$VENV"
    # rdblearn pulls fastdfs (the DFS engine) and relbench 1.1.0; the pin on
    # pandas is fastdfs's -- featuretools/woodwork do not tolerate pandas 3.
    uv pip install --python "$VENV/bin/python" \
        rdblearn fastdfs duckdb pyarrow "pandas<3" scikit-learn loguru
fi
flock -u 9
"$VENV/bin/python" -c "import rdblearn, fastdfs; print('rdblearn', rdblearn.__version__ if hasattr(rdblearn,'__version__') else 'ok')"

# featurize.py imports rt.data (for get_tasks / table_info resolution) from the
# clone, so put the repo's src on the path rather than installing the package into
# the rdblearn env, which would drag relbench-hf back in.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

VERIFY=()
[[ -n ${RT_BUILD_DIR:-} ]] && VERIFY=(--verify-rows "$RT_BUILD_DIR")

"$VENV/bin/python" expts/dbinfer/featurize.py \
    --db "$DB" \
    --pre-dir "$PRE_DIR" \
    "${VERIFY[@]}" \
    "$@"

echo "=== $(date -Is) $DB done ==="
