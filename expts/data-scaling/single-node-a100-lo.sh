#!/bin/bash
# Launch expts/data-scaling/train.py on the 8 A100s of ampere1 under il-lo.
#
# Same shape as single-node.sh, but not under the `il` QOS: `il` caps a100 at 10
# per user and the control run already holds 8, so a second 8-GPU a100 job is
# only reachable via il-lo (priority 100, preemptible -- the requeue machinery
# below covers that).
#
#   ./expts/data-scaling/single-node.sh [extra train.py args...]
#
# Run it from a clean, pushed checkout: the submitter records the repo URL and
# HEAD commit, refuses to submit a dirty or unpushed tree, mints a run id with
# rt.config.timestamp(), and submits itself with sbatch. The job clones the repo
# fresh into a unique /tmp dir, checks out that commit, and runs train.py under
# torchrun with --logger.id=<run id>. The id is fixed at submit time, so every
# requeue (preemption or time limit) reuses the same wandb run and output dir
# and resumes from resume.pt there.
#
# RT_TRAIN_SCRIPT=<path> runs a different entry point (e.g. train_8k.py);
# RT_RESUME_ID=<id> relaunches an existing run instead of starting a new one.

#SBATCH --job-name=rt-data-scaling-a100-lo
#SBATCH --partition=il
#SBATCH --account=infolab
#SBATCH --qos=il-lo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
# Any ampere node with 8 free A100s: they go down and get drained often enough
# that pinning one just makes the job pend. Add --nodelist on the sbatch line to
# force a particular node.
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:a100:8
#SBATCH --ntasks-per-node=1
# Not --exclusive: these nodes usually carry unrelated CPU-only jobs, so
# demanding the whole node would just queue behind them. 112 = the site cap of 14 CPUs per GPU
# on ampere for non-exclusive jobs (8 x 14); asking for more is rejected at
# submit. Memory is left to DefMemPerGPU, which gives 2017232M for 8 GPUs --
# the same allocation the exclusive control run gets.
#SBATCH --cpus-per-task=112
# The submit dir is node-local to the submit node, so don't try to start in it.
#SBATCH --chdir=/tmp
#SBATCH --propagate=MEMLOCK
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@300

set -euo pipefail

LOG_DIR=/dfs/user/ranjanr/slurm-logs/rt-data-scaling-a100-lo

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

    # Run id == wandb id == output subdir; fixed here so requeues resume.
    # RT_RESUME_ID=<id> reuses an existing run instead of minting a new one, so
    # a run that has to be relaunched by hand (not by slurm requeue) picks up
    # its own resume.pt, wandb run and output dir rather than starting over.
    RT_RUN_ID=${RT_RESUME_ID:-$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')}

    mkdir -p "$LOG_DIR"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "run id: $RT_RUN_ID"

    # Deliberately NOT --export=ALL: this shell's env is fish-config'd for the
    # submit node (HOME=/lfs/local/0/..., a node-local PATH) and holds API
    # tokens. Exporting it would point the job at a home that does not exist on
    # the compute node and would stash the tokens in Slurm's job record. Only
    # the RT_* vars travel; the job rebuilds its env below.
    # Slurm env vars outrank #SBATCH directives, so a submit from inside an
    # interactive allocation would silently impose that job's cpu/mem/node
    # shape on this one. Strip them.
    strip=()
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    exec env "${strip[@]}" sbatch \
        --output="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --error="$LOG_DIR/${RT_RUN_ID}_%j.out" \
        --export=RT_REPO="$RT_REPO",RT_COMMIT="$RT_COMMIT",RT_BRANCH="$RT_BRANCH",RT_RUN_ID="$RT_RUN_ID" \
        "$0" "$@"
fi

# ------------------------------------------------------------------- job side
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname), restarts=${SLURM_RESTART_COUNT:-0} ==="
echo "repo=$RT_REPO commit=$RT_COMMIT run_id=$RT_RUN_ID"

# Bootstrap only what the clone itself needs; the full env comes from the
# cloned tree's expts/slurm-env.sh below, so the recorded commit pins the job
# environment too.
export USER=${USER:-$(id -un)}
export TMPDIR=/tmp/$USER
GITHUB_TOKEN=$(tr -d '[:space:]' < "/dfs/user/$USER/.secrets/github")
mkdir -p "$TMPDIR/clones"

WORK_DIR=$(mktemp -d "$TMPDIR/clones/rt-${RT_RUN_ID}-job${SLURM_JOB_ID}.XXXX")
# The pixi env lives inside the clone, so it goes away with it.
trap 'rm -rf "$WORK_DIR"' EXIT
# The -c url.insteadOf rewrite authenticates the fetch without writing the
# token into the clone's .git/config (the remote keeps the plain URL).
git -c url."https://x-access-token:$GITHUB_TOKEN@github.com/".insteadOf="https://github.com/" \
    clone --quiet "$RT_REPO" "$WORK_DIR/relational-transformer"
cd "$WORK_DIR/relational-transformer"
git checkout --quiet "$RT_COMMIT"
echo "clone: $PWD @ $(git rev-parse --short HEAD)"

# One shared definition of the job environment (node-local HOME, PATH, caches,
# tokens); see expts/slurm-env.sh. Nothing env-related is set per job script.
source expts/slurm-env.sh

# Where the ranks write their preemption checkpoint; the trap below waits for
# it. Mirrors rt.train.main: <out_root>/<entity>/<project>/<id>/resume.pt.
RESUME_PT=/dfs/user/ranjanr/ckpts/rtv2/2026-07-24/$RT_RUN_ID/resume.pt

# Static rendezvous with a per-job port (dynamic c10d has wedged under load).
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# pixi.lock is gitignored in this repo, so a fresh clone has none and the first
# job of a run has to solve the environment itself. Keep that solve next to the
# run's logs and reuse it on every requeue, so the environment is identical
# across the whole run instead of re-solving (and drifting) each time.
RUN_LOCK=$LOG_DIR/${RT_RUN_ID}.pixi.lock
if [[ -f $RUN_LOCK ]]; then
    echo "reusing pixi.lock from $RUN_LOCK"
    cp "$RUN_LOCK" pixi.lock
fi
pixi install
[[ -f $RUN_LOCK ]] || cp pixi.lock "$RUN_LOCK"

pixi run build-sampler

# Not plain torchrun: slurm signals every process in the job at preemption,
# so the elastic agent starts killing ranks in the same second they are
# told to save. The shim makes the launcher deaf to SIGTERM; the ranks
# register their own handlers, so they still get it and save first.
pixi run python expts/data-scaling/torchrun_shielded.py \
    --nnodes=1 --nproc-per-node=8 \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "${RT_TRAIN_SCRIPT:-expts/data-scaling/train.py}" \
    --logger.id "$RT_RUN_ID" \
    "$@" &
TRAIN_PID=$!

# ---- preemption ----
# One mechanism, and only one: slurm signals every process in the job, the ranks
# handle it themselves (rt.train.main installs SIGTERM/SIGUSR1 handlers and saves
# resume.pt at the next step boundary), and torchrun_shielded.py keeps the
# elastic agent from killing them before they get there. This script's only job
# is to stay alive until they are done -- if it exited on the signal, slurm would
# tear the step down mid-save.
got_signal=0
on_signal() {
    got_signal=1
    echo "=== $(date -Is) caught $1; waiting for the ranks to save ==="
}
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGUSR1' USR1

resume_before=$( (stat -c %Y "$RESUME_PT" 2>/dev/null) || echo 0 )
# `wait` returns early every time a trap fires, so keep waiting until the child
# is actually gone.
while ! wait "$TRAIN_PID"; do
    kill -0 "$TRAIN_PID" 2>/dev/null || break
done
resume_after=$( (stat -c %Y "$RESUME_PT" 2>/dev/null) || echo 0 )

if (( got_signal )); then
    if [[ $resume_after -gt $resume_before ]]; then
        echo "=== resume.pt saved at $(date -Is -d @"$resume_after") ==="
    else
        echo "WARNING: resume.pt not updated; will resume from the last periodic save" >&2
    fi
    # Preemption requeues the job by itself; a time limit does not, and the two
    # are indistinguishable from here, so ask either way and ignore the error.
    # Not after an operator scancel though -- that would resurrect the job.
    state=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE "JobState=[A-Z]+" | cut -d= -f2)
    if [[ $state == CANCELLED ]]; then
        echo "=== cancelled by operator; not requeueing ==="
    else
        echo "=== requeueing job $SLURM_JOB_ID ==="
        scontrol requeue "$SLURM_JOB_ID" || true
    fi
fi
