# Shared scaffolding for the slurm jobs in expts/ -- source, don't execute.
#
# Every job here has the same skeleton: check the tree is clean and pushed, mint
# a run id, submit itself with sbatch, then (in the job) clone the recorded
# commit into node-local scratch and build the pixi environment. Only the
# resources and the payload differ, so those stay in the per-experiment script
# and everything else lives here.
#
# Submit side:  rt_preflight; rt_submit <log_dir> <sbatch args...> -- "$0" "$@"
# Job side:     (inline clone) then rt_build_env <log_dir>

# Recorded so the job runs exactly the code that was submitted, not whatever
# main happens to be when it starts.
rt_preflight() {
    cd "$(git rev-parse --show-toplevel)" || exit 1

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

    # Run id == wandb id == output subdir, fixed here so that every requeue of
    # the job resumes the same run. RT_RESUME_ID reuses an existing one, for a
    # run that has to be relaunched by hand.
    RT_RUN_ID=${RT_RESUME_ID:-$(pixi run python -c 'from rt.config import timestamp; print(timestamp())')}
    export RT_REPO RT_COMMIT RT_BRANCH RT_RUN_ID
}

# rt_submit <log_dir> <extra sbatch args...> -- <script> <script args...>
rt_submit() {
    local log_dir=$1; shift
    local sbatch_args=()
    while [[ $# -gt 0 && $1 != -- ]]; do sbatch_args+=("$1"); shift; done
    shift  # the --

    mkdir -p "$log_dir"
    echo "repo:   $RT_REPO"
    echo "branch: $RT_BRANCH"
    echo "commit: $RT_COMMIT"
    echo "run id: $RT_RUN_ID"

    # Slurm env vars outrank #SBATCH directives *and* command-line flags from a
    # script submitted inside an allocation, which would silently impose that
    # job's cpu/mem/node shape on this one. Strip them.
    local strip=() k
    while IFS='=' read -r k _; do strip+=(-u "$k"); done < <(env | grep -E '^(SLURM|SBATCH)_')

    # Deliberately not --export=ALL: the submitting shell's env points at a
    # node-local home that does not exist on the compute node, and holds API
    # tokens that would end up in slurm's job record. The job rebuilds its own
    # env from expts/slurm-env.sh; only the RT_* vars travel.
    local exports="RT_REPO=$RT_REPO,RT_COMMIT=$RT_COMMIT,RT_BRANCH=$RT_BRANCH,RT_RUN_ID=$RT_RUN_ID"
    [[ -n ${RT_TRAIN_SCRIPT:-} ]] && exports+=",RT_TRAIN_SCRIPT=$RT_TRAIN_SCRIPT"
    [[ -n ${RT_MODEL:-} ]] && exports+=",RT_MODEL=$RT_MODEL"
    [[ -n ${RT_NPROC:-} ]] && exports+=",RT_NPROC=$RT_NPROC"

    if [[ -n ${RT_DRY_RUN:-} ]]; then
        echo "DRY RUN: sbatch ${sbatch_args[*]} --export=$exports $*"
        exit 0
    fi
    exec env "${strip[@]}" sbatch "${sbatch_args[@]}" --export="$exports" "$@"
}

# rt_build_env <log_dir>: bring the node's env up and build the pixi
# environment. Call it from inside the clone -- the clone itself has to stay
# inline in each job script, since this file only exists once the clone does.
rt_build_env() {
    local log_dir=$1

    # One shared definition of the job environment (node-local HOME, PATH,
    # caches, tokens); see expts/slurm-env.sh.
    source expts/slurm-env.sh

    # pixi.lock is gitignored in this repo, so a fresh clone has none and the
    # first job of a run solves the environment itself. Keep that solve next to
    # the run's logs and reuse it on every requeue, so the environment is
    # identical across the whole run instead of re-solving (and drifting).
    local run_lock=$log_dir/${RT_RUN_ID}.pixi.lock
    if [[ -f $run_lock ]]; then
        echo "reusing pixi.lock from $run_lock"
        cp "$run_lock" pixi.lock
    fi
    pixi install
    [[ -f $run_lock ]] || cp pixi.lock "$run_lock"

    pixi run build-sampler
}
