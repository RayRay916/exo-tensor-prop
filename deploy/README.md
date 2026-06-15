# deploy/ — proportional-TP cluster tooling

Scripts to bring up an isolated, **proportional (memory-weighted) tensor-parallel**
exo cluster across heterogeneous Apple-Silicon nodes. They drive the in-source
sharding added by this fork (see [`../docs/proportional-tp.md`](../docs/proportional-tp.md)).

## Setup

1. Build this fork's venv on **every** node (`uv sync --extra mlx`, see the root
   README), checked out at the same path on each (default `~/exo-tensor-prop`).
2. On the master node:
   ```bash
   cd deploy
   cp cluster.env.example cluster.env     # cluster.env is gitignored
   $EDITOR cluster.env                    # fill in workers, anchors, EXO_BIN, model ids
   ```
3. Make sure the same `cluster.env` (or one with the same ports/namespace) is
   present in each worker's `deploy/` dir — workers source it via SSH.

### Install variants

`EXO_BIN` just has to point at *some* exo built from this fork's code:

- **Standalone fork venv** (clean): build `~/exo-tensor-prop/.venv` per step 1 and
  set `EXO_BIN="$HOME/exo-tensor-prop/.venv/bin/exo"`, `EXO_BIN_RE="exo-tensor-prop/[.]venv/bin/exo"`.
- **Synced into an existing `~/exo`** (no separate build): if you copied the
  fork's source into an existing editable `~/exo` checkout, point
  `EXO_BIN="$HOME/exo/.venv/bin/exo"` and `EXO_BIN_RE="exo/[.]venv/bin/exo"`. The
  scripts can also live in any dir on PATH via a symlink — `exoprop` resolves its
  own real directory, so `cluster.env` is read next to the script, not the symlink.

## Usage

```bash
./exoprop start                 # bring up the mesh, RAM-weighted split (no model)
./exoprop start --weights 128,64,64   # pin the split (rank0..N), in GB or any ratio
./exoprop start 35b             # ensure mesh up, load the MODEL_35B shortcut N-way
./exoprop start load <hf_id>    # load any model N-way
./exoprop status                # mesh + weight-mode + loaded models
./exoprop logs                  # tail the master run.log
./exoprop down                  # stop only this cluster's instance (port-scoped)
./exoprop stop                  # full nuke: kill this cluster's exo on every node
```

## How it works

- **Isolation.** `EXO_HOME`, `EXO_LIBP2P_NAMESPACE`, and dedicated API/libp2p
  ports keep this cluster from colliding with a stock exo cluster on the same
  tailnet.
- **Proportional split.** With `EXO_NODE_WEIGHTS` unset, each node's share of the
  MoE experts is derived from its available RAM; pass `--weights` to pin it.
- **Attention replicated.** `EXO_TP_REPLICATE_ATTN=1` (set by `prop-daemon.sh`)
  replicates the small attention block instead of sharding it — required when
  `num_key_value_heads` doesn't divide across the node count.
- **Laptop bootstrap.** Firewalled laptop workers can't be found via mDNS, so
  each dials an `EXO_ANCHOR_IPS` host to fetch its current libp2p peer-id (exo
  regenerates the peer-id every launch, so a static multiaddr would go stale).

All host/IP/model specifics live in `cluster.env` — nothing machine-specific is
committed to the repo.
