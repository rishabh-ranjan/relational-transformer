#!/usr/bin/env bash
# Preprocess a whole relbench-3.0.0 collection (e.g. the 650-database "Join")
# as a preemptible Slurm array, sharding datasets across array tasks. Each task
# downloads its shard from the Hub, runs the rustler preprocessor + text
# embeddings, and (optionally) uploads the result to a *-preprocessed Hub repo.
#
# This is an INTERNAL convenience for the big run; the preprocessing itself is
# infrastructure-agnostic (see `pixi run preprocess-many`). Discover your
# cluster's accounts/partitions/QoS before launching:
#   sacctmgr -s show user $USER format=account%20,partition%20,qos%60
#   sinfo -o "%P %a %l %D %N"
#
# Usage:
#   NUM_SHARDS=64 OUT_DIR=/dfs/user/$USER/the-join-pre \
#   REPO=stanford-star/the-join UPLOAD_REPO=stanford-star/the-join-preprocessed \
#   sbatch --array=0-63%16 scripts/slurm_preprocess.sh
#
# Re-running is safe: --skip-existing skips datasets whose meta.json is present.

#SBATCH --job-name=join-pre
#SBATCH --output=logs/join-pre-%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
# Set these on the command line or here (left unset so the script is portable):
##SBATCH --account=<acct>
##SBATCH --partition=<preemptible-partition>
##SBATCH --qos=<preemptible-qos>

set -euo pipefail

REPO="${REPO:-stanford-star/the-join}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to a large-storage path, e.g. /dfs/user/$USER/the-join-pre}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L12-v2}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

mkdir -p "$OUT_DIR" logs

# Build the preprocessor binary once (cached afterwards).
pixi run build-pre

UPLOAD_ARGS=()
if [[ -n "${UPLOAD_REPO:-}" ]]; then
  UPLOAD_ARGS=(--upload-repo "$UPLOAD_REPO")
fi

echo "shard $SHARD / $NUM_SHARDS  repo=$REPO  out=$OUT_DIR"
exec pixi run python scripts/preprocess.py many \
  --repo "$REPO" \
  --out-dir "$OUT_DIR" \
  --shard "$SHARD" \
  --num-shards "$NUM_SHARDS" \
  --embedding-model "$EMBEDDING_MODEL" \
  --batch-size "$BATCH_SIZE" \
  --skip-existing \
  "${UPLOAD_ARGS[@]}"
