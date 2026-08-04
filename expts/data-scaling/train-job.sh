#!/bin/bash
# Launch a data-scaling pretraining run under torchrun on one node.
#
#   ./expts/data-scaling/train-job.sh [--profile NAME] [train.py args...]
#
#   --profile ampere      8xA100, `il` QOS (fast queue, 7-day cap)   [default]
#   --profile ampere-lo   8xA100, `il-lo` (when `il`'s 10-a100 cap is spent)
#   --profile blackwell   4xB200 on blackwell1, `il-lo` (b200 is capped at 2
#                         under `il`, so 4 is only reachable here)
#
#   RT_TRAIN_SCRIPT=<path>  run a different entry point (e.g. train_8k.py)
#   RT_RESUME_ID=<id>       relaunch an existing run instead of starting a new
#                           one; it picks up its own resume.pt, wandb run and
#                           output dir rather than starting over
#   RT_DRY_RUN=1            print the sbatch command instead of submitting
#
# Submit from a clean, pushed checkout: the recorded commit is what the job
# clones and runs, and the run id is fixed at submit time so every requeue
# (preemption or time limit) resumes the same run.

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-data-scaling
COMMON=(--partition=il --account=infolab --nodes=1 --ntasks-per-node=1
        # The submit dir is node-local to the submit node, so don't start in it.
        --chdir=/tmp --propagate=MEMLOCK --requeue --open-mode=append)

# ---------------------------------------------------------------- submit side
# RT_RUN_ID (not SLURM_JOB_ID) tells the two sides apart: submitting from inside
# an interactive allocation means SLURM_JOB_ID is already set in your shell.
if [[ -z "${RT_RUN_ID:-}" ]]; then
    profile=ampere
    if [[ ${1:-} == --profile ]]; then profile=$2; shift 2; fi

    case $profile in
    ampere)
        # --exclusive plus the partition's DefMemPerGPU lands on mem=2017232M,
        # the whole node bar its reserve. An explicit --mem is capped lower by
        # MaxMemPerCPU (10700M x 128), and --mem=0 is rejected outright.
        res=(--job-name=rt-ds-ampere --qos=il --time=7-00:00:00
             --constraint=ampere --exclusive --gres=gpu:a100:8 --cpus-per-task=128)
        export RT_NPROC=8 ;;
    ampere-lo)
        # Not --exclusive: these nodes usually carry unrelated CPU-only jobs, so
        # demanding the whole node would just queue behind them. 112 CPUs is the
        # site cap of 14 per GPU for non-exclusive ampere jobs.
        res=(--job-name=rt-ds-ampere-lo --qos=il-lo --time=21-00:00:00
             --constraint=ampere --gres=gpu:a100:8 --cpus-per-task=112)
        export RT_NPROC=8 ;;
    blackwell)
        # Memory is explicit here: the site plugin would default this node to
        # 565G for 4 GPUs, below what the mixture needs in page cache, and
        # 1500000M is the most MaxMemPerCPU allows at 144 CPUs.
        res=(--job-name=rt-ds-blackwell --qos=il-lo --time=21-00:00:00
             --nodelist=blackwell1 --gres=gpu:b200:4 --cpus-per-task=144
             --mem=1500000M)
        export RT_NPROC=4 ;;
    *)
        echo "unknown profile: $profile (ampere | ampere-lo | blackwell)" >&2
        exit 1 ;;
    esac

    source expts/slurm-lib.sh
    rt_preflight
    echo "profile: $profile"
    rt_submit "$LOG_DIR" "${COMMON[@]}" "${res[@]}" \
        --output="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        -- "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID"

# The clone is the one thing that cannot come from the shared library: the
# library lives in the repo, which does not exist on this node yet.
export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
mkdir -p "$TMPDIR/clones"
WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-ds-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
trap 'rm -rf "$WORK_DIR"' EXIT   # the pixi env lives in the clone, so it goes too
git -c url."https://x-access-token:$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

source expts/slurm-lib.sh
rt_build_env "$LOG_DIR"

# Static rendezvous with a per-job port (dynamic c10d has wedged under load).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Not plain torchrun: slurm signals every process in the job at preemption, so
# the elastic agent starts killing ranks in the same second they are told to
# save. The shim makes the launcher deaf to SIGTERM; the ranks register their
# own handlers, so they still get it and save first.
pixi run python expts/data-scaling/torchrun_shielded.py \
    --nnodes=1 --nproc-per-node="$RT_NPROC" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "${RT_TRAIN_SCRIPT:-expts/data-scaling/train.py}" \
    --logger.id "$RT_RUN_ID" \
    "$@" &
TRAIN_PID=$!

# ---- preemption ----
# Slurm's SIGTERM reaches this script, not the ranks: the ranks only ever saw it
# because torchrun's agent forwarded it *while killing them*, which is what
# torchrun_shielded.py deliberately prevents. So the script tells them itself and
# then waits -- they save resume.pt at the next step boundary (atomically, see
# rt.train.main) and exit, and slurm requeues the job (PreemptMode=REQUEUE).
tell_ranks_to_save() {
    echo "=== $(date -Is) caught $1; telling the ranks to save ===" >&2
    local script_base pids p kids args n=0
    script_base=$(basename "${RT_TRAIN_SCRIPT:-expts/data-scaling/train.py}")
    # Walk the whole tree: pixi -> shield -> agent -> ranks is four levels deep,
    # and only the ranks run the train script (the shield's own name contains
    # "torchrun", which is how it is excluded). Dataloader workers are forked
    # from the ranks so they match too; SIGUSR1 is harmless there.
    pids=("$TRAIN_PID")
    while ((${#pids[@]})); do
        p=${pids[0]}; pids=("${pids[@]:1}")
        kids=$(pgrep -P "$p" 2>/dev/null || true)
        [[ -n $kids ]] && pids+=($kids)
        args=$(ps -o args= -p "$p" 2>/dev/null) || continue
        [[ $args == *"$script_base"* && $args != *torchrun* ]] || continue
        kill -USR1 "$p" 2>/dev/null && n=$((n + 1))
    done
    echo "=== signalled $n process(es) ===" >&2
}
trap 'tell_ranks_to_save SIGTERM' TERM
trap 'tell_ranks_to_save SIGUSR1' USR1

# `wait` returns every time a trap fires, so keep waiting for the real exit.
while ! wait "$TRAIN_PID"; do
    kill -0 "$TRAIN_PID" 2>/dev/null || break
done
