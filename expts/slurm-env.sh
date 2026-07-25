# Shared environment for slurm jobs in expts/ -- source, don't execute.
#
# A batch script is not a login shell, so none of the interactive fish config
# runs. This file is the single place that defines what a job's env looks like,
# so no individual job script has to get it right.
#
# It does not paper over an unprepared node: verify-node.sh (from the dotfiles
# repo) checks that the node meets the standard assumptions -- a node-local home
# that is a dotfiles checkout, with a node-local pixi and the global CLI tools --
# and sets the node up if it does not. Everything below then just uses them.
#
# No fallbacks: anything that cannot be reinstated is a hard error here, seconds
# into the job, rather than a silently degraded run discovered hours later.

set -euo pipefail

die() { echo "slurm-env: $*" >&2; exit 1; }

export USER=${USER:-$(id -un)}
export HOME=/lfs/local/0/$USER
SETUP_URL=https://raw.githubusercontent.com/rishabh-ranjan/dotfiles/main/setup-node.sh

if [[ -f $HOME/verify-node.sh ]]; then
    bash "$HOME/verify-node.sh" || die "node not usable"
else
    # Node has never been touched: bootstrap it, then verify.
    curl -fsSL "$SETUP_URL" | bash || die "could not set up $(hostname -s)"
    bash "$HOME/verify-node.sh" || die "node not usable"
fi

# pixi lives in the node-local home, and so do the environments it builds
# (detached-environments is off, so each env sits inside its project, and
# projects are node-local clones under $TMPDIR).
export PIXI_HOME=$HOME/.pixi
export PATH=$PIXI_HOME/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # points into a home that is not this job's home
export TMPDIR=/tmp/$USER
export XDG_CACHE_HOME=$HOME/.cache
export WANDB_DIR=$HOME/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# Tokens come from the shared secrets dir rather than the job env, where slurm
# would record them. verify-node.sh has already checked they are readable.
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

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
ulimit -l unlimited || die "cannot raise RLIMIT_MEMLOCK (need --propagate=MEMLOCK)"

echo "slurm-env: host=$(hostname -s) HOME=$HOME pixi=$(pixi --version)"
