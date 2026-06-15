# Proportional (memory-weighted) tensor parallelism

## The problem

Stock MLX tensor parallelism splits every weight matrix into `group.size()`
**equal** shards. On a heterogeneous cluster that wastes RAM: a 128 GB Mac Studio
gets the same shard as a 64 GB MacBook Pro, so the whole cluster is bounded by
its smallest node. You either leave the big node's memory idle or you can't fit
the model at all.

This fork shards the **MoE expert weights** — which are ~96% of the parameters in
the target models — *proportionally* to each node's GPU budget, while leaving the
small attention block **replicated**. The result: a big node carries a bigger
slice, so the cluster's capacity is the *sum* of node budgets rather than
`min(node) × node_count`.

## Why this is safe with uneven shards

The only cross-node collective on the MoE hot path is `all_sum` (added by exo's
existing `ShardedMoE` wrapper), which is **size-agnostic**. Uneven shard sizes
are therefore inherently safe — no all-gather of ragged tensors is ever required.

## Why attention is replicated, not sharded

Both initial target models (Qwen3-Next-80B-A3B, Qwen3.5-122B-A10B) have
`num_key_value_heads == 2`. Two KV heads cannot be divided across three nodes,
and the attention/KV footprint is tiny next to the experts — so attention + KV
cache are replicated. Set `EXO_TP_REPLICATE_ATTN=1` (the deploy daemon sets this)
to take that path.

## Where it lives in the source

| Concern | Location | Key symbol |
|---|---|---|
| Per-rank weight vector from node memory / `EXO_NODE_WEIGHTS` | `src/exo/master/placement_utils.py` | `_tensor_proportions`, `allocate_layers_proportionally` |
| Carrying proportions to the worker | `src/exo/shared/types/worker/shards.py` | `TensorShardMetadata.proportions` |
| Group-size-aligned integer split of each weight dim | `src/exo/worker/engines/mlx/auto_parallel.py` | `_aligned_segment_sizes` |
| Applying the split + replicating attention | `src/exo/worker/engines/mlx/auto_parallel.py` | `tensor_auto_parallel(model, group, proportions)` |
| Memory budget used for the split | `src/exo/utils/info_gatherer/macmon.py` | `OVERRIDE_MEMORY_MB` hook |

Design details (from the implementation):

* Splits use a **largest-remainder** algorithm in units of `align` so quantized
  group boundaries (`group_size`, default 64) stay intact.
* Output-dim slicing (gate/up, "all-to-sharded") has no alignment constraint.
* Input-dim slicing (down, "sharded-to-all") must land on `group_size`
  boundaries; the packed-uint32 axis is sliced by the pack factor inferred from
  array shapes (no dependency on a `.bits` attribute).
* If the computed split is (near) even, proportions are left unset so workers
  take the stock equal-split path — zero behavior change on homogeneous clusters.

## Controlling the split

**Automatic (default):** with `EXO_NODE_WEIGHTS` unset, each node's share is
derived from its available RAM (via macmon, or the psutil fallback).

**Pinned:** set `EXO_NODE_WEIGHTS` to override. Accepts rank keys or a bare ratio:

```bash
EXO_NODE_WEIGHTS='rank0:0.80,rank1:0.20'   # explicit fractions
EXO_NODE_WEIGHTS='128,64,64'               # GB (or any ratio); normalized
```

`deploy/exoprop start --weights 128,64,64` is sugar for the second form.

**Biasing a node's reported memory:** `OVERRIDE_MEMORY_MB` caps a node's
reported `ram_available` on the macmon path, so you can hold memory back for KV
cache / headroom without disabling macmon.

## Related runtime knobs

| Env var | Effect |
|---|---|
| `EXO_TP_REPLICATE_ATTN=1` | Replicate attention instead of sharding it (required when KV heads don't divide the node count). |
| `EXO_NODE_WEIGHTS` | Pin the per-rank split; unset → derive from RAM. |
| `OVERRIDE_MEMORY_MB` | Cap a node's reported available RAM (bias the split). |
| `EXO_KV_BITS` | Opt-in KV-cache quantization (e.g. `8`); unset → off, behavior unchanged. |
| `EXO_MAX_CONCURRENT_REQUESTS` | Continuous-batch concurrency per instance. |

See [`../deploy/README.md`](../deploy/README.md) for bringing up a cluster.
