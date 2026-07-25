# Shared environment for slurm jobs in expts/ -- source, don't execute.
#
# A batch script is not a login shell, so none of the interactive fish config
# runs. This file is the single place that defines what a job's env looks like,
# so no individual job script has to get it right. It is node-agnostic: every
# path is derived on the machine it runs on, so it behaves the same on the dev
# node and on any compute node.

export USER=${USER:-$(id -un)}

# Node-local home: pixi envs, HF/torch caches, wandb staging. Falls back to the
# shared home if this node has no /lfs/local scratch for us.
_node_home=/lfs/local/0/$USER
mkdir -p "$_node_home" 2>/dev/null || _node_home=/sailhome/$USER
export HOME=$_node_home
export PATH=$HOME/.pixi/bin:/sailhome/$USER/.pixi/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
unset PYTHONPATH  # a dev-node-local dir; would shadow the pixi env
export TMPDIR=/tmp/$USER
export XDG_CACHE_HOME=$HOME/.cache
export WANDB_DIR=$HOME/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# Tokens live in the shared secrets dir (readable from every node) rather than
# being exported into the job env, where slurm would record them.
_secrets=/sailhome/$USER/.secrets
_read_secret() { tr -d '[:space:]' < "$_secrets/$1"; }
WANDB_API_KEY=$(_read_secret wandb); export WANDB_API_KEY
HF_TOKEN=$(_read_secret huggingface); export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
GITHUB_TOKEN=$(_read_secret github); export GITHUB_TOKEN GH_TOKEN=$GITHUB_TOKEN

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
ulimit -l unlimited 2>/dev/null || true

echo "slurm-env: host=$(hostname -s) HOME=$HOME pixi=$(command -v pixi || echo MISSING)"
