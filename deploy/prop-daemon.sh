#!/bin/bash
# exo-tensor-prop node daemon — exo's proportional (memory-weighted) tensor
# sharding. Isolation via EXO_HOME + EXO_LIBP2P_NAMESPACE + dedicated ports.
# All infra comes from deploy/cluster.env (see cluster.env.example).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/cluster.env" ]; then . "$HERE/cluster.env"; else
  echo "prop-daemon: $HERE/cluster.env missing (copy cluster.env.example -> cluster.env)"; exit 1
fi

export EXO_HOME="${EXO_PROP_HOME:-.exo-prop}"
export EXO_LIBP2P_NAMESPACE="${EXO_PROP_NAMESPACE:-prop-3node}"
export EXO_MODELS_READ_ONLY_DIRS="${EXO_MODELS_READ_ONLY_DIRS:-$HOME/.exo/models}"
export MLX_METAL_FAST_SYNCH=1
export EXO_TP_REPLICATE_ATTN=1          # replicate attention; shard MoE experts only
export EXO_OFFLINE=true
mkdir -p "$HOME/$EXO_HOME/models" "$HOME/$EXO_HOME/custom_model_cards"

# Firewalled laptop workers can't be found via mDNS, so each dials an anchor to
# fetch its CURRENT libp2p peer-id (exo regenerates the peer-id every launch, so
# a static multiaddr would go stale). Override anchors via EXO_ANCHOR_IPS.
case "$(scutil --get ComputerName 2>/dev/null)" in
  *MacBook*|*Laptop*)
    if [ -z "${EXO_BOOTSTRAP_PEERS:-}" ]; then
      for _anchor in ${EXO_ANCHOR_IPS:-}; do
        for _i in $(seq 1 20); do
          _pid=$(curl -fsS --max-time 3 "http://$_anchor:${EXO_PROP_API_PORT}/node_id" 2>/dev/null | tr -d '"[:space:]')
          case "$_pid" in
            12D3*) export EXO_BOOTSTRAP_PEERS="/ip4/$_anchor/tcp/${EXO_PROP_LIBP2P_PORT}/p2p/$_pid"
                   echo "exo-prop: bootstrap -> $EXO_BOOTSTRAP_PEERS"; break 2 ;;
          esac
          sleep 2
        done
      done
      [ -z "${EXO_BOOTSTRAP_PEERS:-}" ] && echo "exo-prop: WARN no anchor reachable; launching unbootstrapped"
    fi
    ;;
esac

exec "${EXO_BIN:?set EXO_BIN in cluster.env}" \
  --api-port "${EXO_PROP_API_PORT}" --libp2p-port "${EXO_PROP_LIBP2P_PORT}" "$@" \
  > "$HOME/$EXO_HOME/run.log" 2>&1
