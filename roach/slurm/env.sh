# Shared environment for slurm jobs in expts/ -- source, don't execute.
#
# A batch script is not a login shell, so none of the interactive fish config
# runs. This file is the single place that defines what a job's env looks like,
# so no individual job script has to get it right.
#
# It does not paper over an unprepared node: it sources .bashrc.user, the same
# entry point an interactive login uses, which runs setup-node.sh -- a node-local
# home that is a dotfiles checkout, with a node-local pixi and the global CLI
# tools. Without --update, so it never pulls a node out from under a running job.
#
# No fallbacks: anything that cannot be reinstated is a hard error here, seconds
# into the job, rather than a silently degraded run discovered hours later.

set -euo pipefail

die() { echo "slurm-env: $*" >&2; exit 1; }

export USER=${USER:-$(id -un)}
SETUP_URL=https://raw.githubusercontent.com/rishabh-ranjan/dotfiles/main/setup-node.sh

# Go through the same entry point an interactive login does -- .bashrc.user in
# the real (passwd) home -- so bringing a node up to spec has exactly one path.
# It runs setup-node.sh and switches HOME to the node-local dotfiles checkout;
# sourcing it non-interactively stops before the fish exec. It also cd's to the
# new HOME, hence the restore.
_passwd_home=$(getent passwd "$USER" | cut -d: -f6)
_cwd=$PWD
if [[ -f $_passwd_home/.bashrc.user ]]; then
    source "$_passwd_home/.bashrc.user"
else
    curl -fsSL "$SETUP_URL" | bash || die "could not set up $(hostname -s)"
    export HOME=/lfs/local/0/$USER
fi
cd "$_cwd"

[[ -d $HOME/.git && -x $HOME/.pixi/bin/pixi ]] ||
    die "$(hostname -s) not set up (HOME=$HOME); see $SETUP_URL"

# Shared state that no amount of node setup can create.
for s in wandb huggingface github; do
    [[ -r /dfs/user/$USER/.secrets/$s ]] ||
        die "missing secret /dfs/user/$USER/.secrets/$s"
done

# pixi lives in the node-local home, and so do the environments it builds
# (keyed on the project path; projects are node-local clones).
export PIXI_HOME=$HOME/.pixi
export PATH=$PIXI_HOME/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # points into a home that is not this job's home
# The node-local NVMe, not /tmp. /tmp is the root filesystem -- a few hundred
# GB shared with the OS -- and one relarena export is ~85G, so a handful of them
# fill it and wedge every job on the node. This is also where a run's resume
# point lives, so it has to survive a requeue on a disk with room for it.
export TMPDIR=/lfs/local/0/$USER/tmp
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$HOME/.cache
# One cargo target dir for every clone on this node. Clones are per commit, so
# a per-clone target dir compiles the crate's whole dependency tree for every
# commit, whether or not it touched rust; shared, a commit that changed no rust
# reuses it -- measured, a second checkout of the same commit went from 3m39s to
# 15s. This only works together with the linker pinned in pyproject.toml
# ([tool.pixi.activation.env]): without it cargo's fingerprint depends on the
# environment's path and everything rebuilds anyway, which is what made an
# earlier attempt at this look useless. Cargo locks the directory, so two builds
# at once queue rather than corrupt it.
export CARGO_TARGET_DIR=$XDG_CACHE_HOME/cargo-target
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CARGO_TARGET_DIR"

# Tokens come from the shared secrets dir rather than the job env, where slurm
# would record them; they were checked for readability above.
_secrets=/dfs/user/$USER/.secrets
_read_secret() {
    local v
    v=$(tr -d '[:space:]' < "$_secrets/$1")
    [[ -n $v ]] || die "empty secret $_secrets/$1"
    printf '%s' "$v"
}
WANDB_API_KEY=$(_read_secret wandb); export WANDB_API_KEY
HF_TOKEN=$(_read_secret huggingface); export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
GITHUB_TOKEN=$(_read_secret github); export GITHUB_TOKEN GH_TOKEN=$GITHUB_TOKEN

# No OMP_NUM_THREADS: TaskPlugin is task/cgroup,task/affinity, so slurm binds
# each task to its own cpus_per_task and OpenMP already sizes its pool from that
# mask (measured: nproc is 8 in a --cpus-per-task=8 task on an 80-cpu node). A
# number here could only disagree with the allocation -- as the old default of 8
# did, on every job that asked for more.
export TOKENIZERS_PARALLELISM=false
ulimit -l unlimited || die "cannot raise RLIMIT_MEMLOCK (need --propagate=MEMLOCK)"

echo "slurm-env: host=$(hostname -s) HOME=$HOME pixi=$(pixi --version)"
