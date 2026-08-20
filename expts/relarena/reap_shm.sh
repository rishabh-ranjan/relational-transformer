#!/usr/bin/env bash
# Delete /dev/shm segments whose owning process is gone.
#
# rt shares worker tensors by descriptor since v1.6.0, so a current run leaks
# nothing. This clears what older runs left behind, and is cheap insurance
# against any other tenant on the node doing the same. Only segments owned by
# this user, and only those whose pid is dead, are touched.
set -u
n=0
for f in /dev/shm/torch_* /dev/shm/pymp-*; do
  [ -e "$f" ] || continue
  [ -O "$f" ] || continue                      # ours only
  p=${f#/dev/shm/torch_}; p=${p%%_*}
  case "$p" in ([0-9]*) ;; (*) continue ;; esac
  kill -0 "$p" 2>/dev/null || { rm -rf "$f" 2>/dev/null && n=$((n+1)); }
done
echo "reap_shm: removed $n orphaned segments; $(df -h /dev/shm | awk 'NR==2{print $4" free"}')"
