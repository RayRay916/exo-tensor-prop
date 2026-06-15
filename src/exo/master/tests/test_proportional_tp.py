import pytest

from exo.master.placement_utils import (
    _parse_node_weights_env,  # pyright: ignore[reportPrivateUsage]
    _tensor_proportions,  # pyright: ignore[reportPrivateUsage]
    get_shard_assignments_for_tensor_parallel,
)
from exo.master.tests.conftest import create_node_memory
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Cycle
from exo.shared.types.worker.shards import TensorShardMetadata


def _approx(actual: list[float], expected: list[float], tol: float = 1e-6) -> bool:
    return len(actual) == len(expected) and all(
        abs(a - e) <= tol for a, e in zip(actual, expected, strict=True)
    )


def _model_card(storage_kb: int = 1000) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        n_layers=8,
        storage_size=Memory.from_kb(storage_kb),
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


# ---- _parse_node_weights_env -------------------------------------------------


def test_parse_weights_rank_and_uuid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_NODE_WEIGHTS", "rank0:0.8, abc:0.2 , bad, x:notnum")
    parsed = _parse_node_weights_env()
    assert parsed == {"rank0": 0.8, "abc": 0.2}


def test_parse_weights_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    assert _parse_node_weights_env() == {}


# ---- _tensor_proportions -----------------------------------------------------


def test_proportions_from_rank_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_NODE_WEIGHTS", "rank0:3,rank1:1")
    nodes = [NodeId(), NodeId()]
    mem = {n: create_node_memory(100) for n in nodes}
    assert _approx(_tensor_proportions(nodes, mem), [0.75, 0.25])


def test_proportions_uuid_substring(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_NODE_WEIGHTS", "alpha:9,beta:1")
    nodes = [NodeId("node-alpha"), NodeId("node-beta")]
    mem = {n: create_node_memory(100) for n in nodes}
    assert _approx(_tensor_proportions(nodes, mem), [0.9, 0.1])


def test_proportions_ram_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    nodes = [NodeId(), NodeId()]
    mem = {nodes[0]: create_node_memory(300), nodes[1]: create_node_memory(100)}
    assert _approx(_tensor_proportions(nodes, mem), [0.75, 0.25])


def test_proportions_zero_total_falls_back_even(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_NODE_WEIGHTS", "rank0:0,rank1:0")
    nodes = [NodeId(), NodeId()]
    mem = {n: create_node_memory(0) for n in nodes}
    assert _approx(_tensor_proportions(nodes, mem), [0.5, 0.5])


# ---- memory validation + even-snap (get_shard_assignments_for_tensor_parallel)


def test_even_split_leaves_proportions_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    nodes = [NodeId(), NodeId()]
    cycle = Cycle(node_ids=nodes)
    # Equal RAM -> within EXO_TP_EVEN_TOL -> even path (proportions None).
    mem = {n: create_node_memory(Memory.from_kb(2000).in_bytes) for n in nodes}
    assignments = get_shard_assignments_for_tensor_parallel(
        _model_card(), cycle, mem
    )
    shards = list(assignments.runner_to_shard.values())
    assert all(isinstance(s, TensorShardMetadata) for s in shards)
    assert all(
        s.proportions is None
        for s in shards
        if isinstance(s, TensorShardMetadata)
    )


def test_near_even_snaps_to_even(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    nodes = [NodeId(), NodeId()]
    cycle = Cycle(node_ids=nodes)
    # 1% RAM jitter is within the default 2% tolerance -> still even.
    mem = {
        nodes[0]: create_node_memory(Memory.from_kb(2020).in_bytes),
        nodes[1]: create_node_memory(Memory.from_kb(1980).in_bytes),
    }
    assignments = get_shard_assignments_for_tensor_parallel(
        _model_card(), cycle, mem
    )
    shards = [
        s
        for s in assignments.runner_to_shard.values()
        if isinstance(s, TensorShardMetadata)
    ]
    assert all(s.proportions is None for s in shards)


def test_uneven_sets_proportions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    nodes = [NodeId(), NodeId()]
    cycle = Cycle(node_ids=nodes)
    mem = {
        nodes[0]: create_node_memory(Memory.from_kb(3000).in_bytes),
        nodes[1]: create_node_memory(Memory.from_kb(1000).in_bytes),
    }
    assignments = get_shard_assignments_for_tensor_parallel(
        _model_card(), cycle, mem
    )
    shards = [
        s
        for s in assignments.runner_to_shard.values()
        if isinstance(s, TensorShardMetadata)
    ]
    assert all(s.proportions is not None for s in shards)


def test_insufficient_memory_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_NODE_WEIGHTS", raising=False)
    monkeypatch.setenv("EXO_TP_MEM_HEADROOM", "1.0")
    nodes = [NodeId(), NodeId()]
    cycle = Cycle(node_ids=nodes)
    # storage 1000 KB total; node1 gets ~half but only has room for a sliver.
    mem = {
        nodes[0]: create_node_memory(Memory.from_kb(600).in_bytes),
        nodes[1]: create_node_memory(Memory.from_kb(10).in_bytes),
    }
    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments_for_tensor_parallel(_model_card(1000), cycle, mem)


def test_replicate_fraction_biases_required_up(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_NODE_WEIGHTS", "rank0:99,rank1:1")
    monkeypatch.setenv("EXO_TP_MEM_HEADROOM", "1.0")
    monkeypatch.setenv("EXO_TP_REPLICATE_FRACTION", "0.5")
    nodes = [NodeId(), NodeId()]
    cycle = Cycle(node_ids=nodes)
    # Tiny node's pure proportional share (~1%) of 1000KB would fit in 10KB, but
    # with replicate_attention the 50% replicated baseline pushes it over.
    mem = {
        nodes[0]: create_node_memory(Memory.from_kb(2000).in_bytes),
        nodes[1]: create_node_memory(Memory.from_kb(10).in_bytes),
    }
    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments_for_tensor_parallel(
            _model_card(1000), cycle, mem, replicate_attention=True
        )
