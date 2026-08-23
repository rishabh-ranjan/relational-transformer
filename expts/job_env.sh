# This repo's per-job environment: sourced by roach after the cluster's own
# (see roach.slurm's README), on every node a job holds, before the clone is
# built. What a job of this project needs that is not the cluster's business.

# One cargo target dir for every clone on this node. Clones are per commit, so
# a per-clone target dir compiles the crate's whole dependency tree for every
# commit, whether or not it touched rust; shared, a commit that changed no rust
# reuses it (measured: a second checkout of the same commit went from 3m39s to
# 15s). Only works together with the linker pinned in pyproject.toml
# ([tool.pixi.activation.env]): without it cargo's fingerprint depends on the
# environment's path and everything rebuilds anyway. Cargo locks the directory,
# so two builds at once queue rather than corrupt it.
export CARGO_TARGET_DIR=$XDG_CACHE_HOME/cargo-target
mkdir -p "$CARGO_TARGET_DIR"

# Segments left by a killed DataLoader worker are never reclaimed, and a node
# that fills /dev/shm wedges every job on it. Clear what dead processes of ours
# left behind.
for f in /dev/shm/torch_*; do
    [ -e "$f" ] && [ -O "$f" ] || continue
    p=${f#/dev/shm/torch_}; p=${p%%_*}
    case "$p" in ([0-9]*) kill -0 "$p" 2>/dev/null || rm -f "$f" ;; esac
done
echo "job-env: /dev/shm $(df -h /dev/shm | awk 'NR==2{print $4}') free"

export TOKENIZERS_PARALLELISM=false
# A hung rank can be made to print every thread's Python stack into the job
# log: `srun --jobid=<id> --overlap -N1 -n1 kill -ABRT <rank pid>` (it dies).
export PYTHONFAULTHANDLER=1
# The mixture is mlocked (examples/mlock.py); roach passes --propagate=MEMLOCK.
ulimit -l unlimited || { echo "job-env: cannot raise RLIMIT_MEMLOCK" >&2; exit 1; }
