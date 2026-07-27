#!/bin/bash
# Submit the whole 4DBInfer campaign at once, chained by slurm dependencies.
#
#   ./expts/dbinfer/launch.sh
#   RT_SKIP_FEATURIZE=1 ./expts/dbinfer/launch.sh    # rt-j path only
#
# Nothing waits for anything else at *submit* time: every stage goes into the queue
# now and starts the moment its input exists. That is the fastest route to a first
# number, which is the point -- rather than running stage 1, coming back, running
# stage 2, and so on.
#
#   stage 1  preprocess     array of 4 (one per db)   il-lo, non-ampere GPU node
#   stage 2  featurize      array of 4, per-db dep    il-lo, non-ampere CPU-only
#   stage 3  eval rt        after all of stage 1      il,    8x a100 (not ampere4)
#   stage 3  eval rdblearn  after all of stage 2      il-lo, 8x a100 (not ampere4)
#
# Stage 2 is chained *per database* (``afterok:<arrayjob>_<idx>``), so a database
# starts its DFS as soon as its own preprocess finishes instead of waiting for the
# slowest of the four.
#
# The QOS split is forced by the cluster: the `il` QOS caps a user at
# gres/gpu:a100=10, which is one 8-GPU node. RT-J is the priority, so it takes `il`
# and the baseline takes `il-lo`.
#
# Re-running is safe and is the intended way to top up: every stage skips work whose
# output already exists, so a rerun after a failure submits only what is missing.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
HERE=expts/dbinfer

if [[ -n "$(git status --porcelain)" ]]; then
    echo "WARNING: working tree is dirty; commit or stash before submitting." >&2
    git status --short >&2
    exit 1
fi

NDBS=3  # keep in step with DBS in the stage scripts

echo "=== stage 1: preprocess"
PRE_ARGS=(); [[ -n ${RT_PRE_ARGS:-} ]] && read -r -a PRE_ARGS <<<"$RT_PRE_ARGS"
PRE_OUT=$("$HERE/slurm_preprocess.sh" "${PRE_ARGS[@]+"${PRE_ARGS[@]}"}")
echo "$PRE_OUT"
PRE_JOB=$(grep -oE '[0-9]+$' <<<"$(grep 'Submitted batch job' <<<"$PRE_OUT")")
[[ -n $PRE_JOB ]] || { echo "could not parse the preprocess job id" >&2; exit 1; }

if [[ -z ${RT_SKIP_FEATURIZE:-} ]]; then
    echo
    echo "=== stage 2: featurize (chained per database)"
    FEAT_JOBS=()
    for i in $(seq 0 $((NDBS - 1))); do
        # afterok on this database's own array task, not the whole array.
        OUT=$(RT_DEPEND="afterok:${PRE_JOB}_${i}" RT_ONLY_INDEX="$i" \
              "$HERE/slurm_featurize.sh")
        echo "$OUT"
        FEAT_JOBS+=("$(grep -oE '[0-9]+$' <<<"$(grep 'Submitted batch job' <<<"$OUT")")")
    done
fi

echo
echo "=== stage 3: eval rt (il, 8x a100)"
RT_QOS=il RT_DEPEND="afterok:$PRE_JOB" RT_METHOD=rt "$HERE/slurm_eval.sh"

if [[ -z ${RT_SKIP_FEATURIZE:-} ]]; then
    echo
    echo "=== stage 3: eval rdblearn_tabicl (il-lo, 8x a100)"
    DEP=$(IFS=:; echo "afterok:${FEAT_JOBS[*]}")
    RT_QOS=il-lo RT_DEPEND="$DEP" RT_METHOD=rdblearn_tabicl "$HERE/slurm_eval.sh"
fi

echo
echo "queued:"
squeue -u "$USER" -o "%.10i %.14P %.22j %.9q %.8T %.10M %R" | grep -E "JOBID|dbinfer" || true
echo
echo "collect with:"
echo "  pixi run python $HERE/reduce.py --out-dir /dfs/user/$USER/dbinfer-scaling --per-task"
