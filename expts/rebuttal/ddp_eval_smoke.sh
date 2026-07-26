#!/bin/bash
# Multi-GPU smoke test for `rt.cli.eval` under torchrun.
#
# Runs one small task (rel-f1/driver-top3, test split) twice on real GPUs --
# once single-process, once as 2-rank DDP -- and asserts the submission CSVs
# match. Rank sharding changes only which rank builds which row's context, so
# the predictions must be identical row-for-row.
#
#   sbatch --account=<acct> expts/rebuttal/ddp_eval_smoke.sh
#
# Env: REF (git ref to test, default main), SRC_PRE (preprocessed data to
# stage node-local).
#SBATCH -p il
#SBATCH --gres=gpu:a100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 60
#SBATCH -J ddp-eval-smoke
#SBATCH -o /dfs/user/ranjanr/slurm-logs/ddp-eval-smoke-%j.out
set -euo pipefail

REF="${REF:-main}"
SRC_PRE="${SRC_PRE:-/dfs/user/ranjanr/pre/relbench-preprocessed}"
CKPT="${CKPT:-stanford-star/rt-j/classification}"
DB=rel-f1
TASK=driver-top3

# Everything the job touches lives on node-local disk: /dfs is shared but slow,
# and /lfs/<other-host> is not mounted on the GPU nodes at all.
REPO=/tmp/ranjanr/clones/rt-ddp-smoke
PRE=/tmp/ranjanr/rt-ddp-smoke/pre
WORK=/tmp/ranjanr/rt-ddp-smoke

nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -ge 2 ] || { echo "need >=2 GPUs, got $NGPU"; exit 1; }

rm -rf "$REPO" "$WORK"
mkdir -p "$(dirname "$REPO")" "$PRE"

git clone -q https://github.com/rishabh-ranjan/relational-transformer.git "$REPO"
cd "$REPO"
git checkout -q "$REF"
echo "testing $(git log --oneline -1)"

# Stage only the one database this smoke test needs (~13M).
cp -r "$SRC_PRE/$DB" "$PRE/"
echo "[[\"$DB\", \"$TASK\"]]" > "$WORK/tasks.json"

pixi run -q build-sampler

COMMON=(
  --model.load-ckpt-path "$CKPT"
  --eval.db-task-list "$WORK/tasks.json"
  --eval.pre-dir "$PRE"
  --eval.splits test
  --eval.num-workers 4
)

# NB: do not scope the GPUs with CUDA_VISIBLE_DEVICES here -- exporting it
# around `pixi run` lands after CUDA init and silently drops the run to CPU
# (~1000s/batch). Slurm already scopes the job; the single-process run just
# takes cuda:0.
echo "=== 1 process, 1 GPU ==="
time pixi run -- python -m rt.cli.eval "${COMMON[@]}" --eval.csv-out-dir "$WORK/out1"

echo "=== torchrun, 2 GPUs (DDP) ==="
time pixi run -- torchrun --nproc-per-node=2 --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  -m rt.cli.eval "${COMMON[@]}" --eval.csv-out-dir "$WORK/out2"

echo "=== comparing submissions ==="
pixi run -- python - "$WORK/out1" "$WORK/out2" <<'PY'
import glob, os, sys

import pandas as pd

one, two = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(one, "**", "*.csv"), recursive=True))
assert files, f"no submission CSVs under {one}"
bad = []
for f in files:
    g = os.path.join(two, os.path.relpath(f, one))
    if not os.path.exists(g):
        bad.append(f"{g}: missing in the DDP run")
        continue
    # Row order is not part of the submission's meaning: rows are keyed by
    # their entity/timestamp columns, which DDP interleaves by rank.
    a, b = pd.read_csv(f), pd.read_csv(g)
    keys = [c for c in a.columns if a[c].dtype.kind in "iuOM"]
    a, b = (df.sort_values(keys).reset_index(drop=True) for df in (a, b))
    if a.shape != b.shape:
        bad.append(f"{f}: shape {a.shape} vs {b.shape}")
        continue
    diff = (a.select_dtypes("number") - b.select_dtypes("number")).abs().max().max()
    print(f"{os.path.relpath(f, one)}: rows={len(a)} max_abs_diff={diff:.3g}")
    if diff > 1e-4:
        bad.append(f"{f}: predictions differ by {diff}")
if bad:
    print("FAIL:", *bad, sep="\n  ")
    sys.exit(1)
print("OK: 1-GPU and 2-GPU DDP submissions agree")
PY

echo "smoke test passed"
