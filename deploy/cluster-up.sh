#!/bin/bash
# exo-tensor-prop N-node bring-up. Run INSIDE tmux via the `exoprop` wrapper,
# which adds teardown + persistence. Master (rank0) is this host; workers are
# reached over held SSH connections.
#
# Optional: --weights SPEC sets EXO_NODE_WEIGHTS to pin the split (e.g.
# "rank0:128,rank1:64,rank2:64"); unset => the split is derived from each node's
# available RAM by the proportional sharder.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/cluster.env" ]; then . "$HERE/cluster.env"; else
  echo "cluster-up: $HERE/cluster.env missing (copy cluster.env.example -> cluster.env)"; exit 1
fi

export EXO_MAX_CONCURRENT_REQUESTS="${EXO_MAX_CONCURRENT_REQUESTS:-4}"
if [ "${1:-}" = "--weights" ] && [ -n "${2:-}" ]; then export EXO_NODE_WEIGHTS="$2"; fi
WPREFIX=""
[ -n "${EXO_NODE_WEIGHTS:-}" ] && WPREFIX="EXO_NODE_WEIGHTS='$EXO_NODE_WEIGHTS' "

# rank0 (master)
bash "$HERE/prop-daemon.sh" -m &
sleep 10
# rank1..N (workers, over held SSH)
RDIR="${EXO_REMOTE_DEPLOY_DIR:-$HOME/exo-tensor-prop/deploy}"
for h in ${EXO_WORKERS:-}; do
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=1000 "$h" \
    "EXO_MAX_CONCURRENT_REQUESTS=$EXO_MAX_CONCURRENT_REQUESTS ${WPREFIX}bash $RDIR/prop-daemon.sh" &
done
wait
