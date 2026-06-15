import pytest

from exo.worker.engines.mlx.sharding_math import (
    aligned_segment_sizes,
    shard_axis_bounds,
)


def test_segments_sum_to_dim_and_align():
    sizes = aligned_segment_sizes(dim=768, proportions=[0.7, 0.3], align=64)
    assert sum(sizes) == 768
    assert all(s % 64 == 0 for s in sizes)
    # Larger proportion gets the larger shard.
    assert sizes[0] > sizes[1]


def test_even_proportions_split_evenly():
    sizes = aligned_segment_sizes(dim=1024, proportions=[0.25] * 4, align=64)
    assert sizes == [256, 256, 256, 256]


def test_min_one_unit_no_empty_shard():
    # Rank 1/2 would round to zero units; each must still get one align unit.
    sizes = aligned_segment_sizes(dim=192, proportions=[0.98, 0.01, 0.01], align=64)
    assert sum(sizes) == 192
    assert all(s >= 64 for s in sizes)


def test_align_one_unquantized():
    sizes = aligned_segment_sizes(dim=10, proportions=[0.5, 0.5], align=1)
    assert sizes == [5, 5]


def test_raises_when_units_below_ranks():
    # 2 group-units cannot cover 3 ranks.
    with pytest.raises(ValueError):
        aligned_segment_sizes(dim=128, proportions=[0.4, 0.3, 0.3], align=64)


def test_dim_not_multiple_of_align_asserts():
    with pytest.raises(AssertionError):
        aligned_segment_sizes(dim=100, proportions=[0.5, 0.5], align=64)


def test_axis_bounds_integer_and_consistent():
    # intermediate=768, group_size=64 -> rank0 holds units [0, 512).
    total = 768
    # Packed 4-bit weight axis (pack factor 8) and scales axis (group 64) differ
    # in length but must slice at the same logical fraction.
    w_start, w_end = shard_axis_bounds(0, 512, total, 768 // 8)  # 96
    s_start, s_end = shard_axis_bounds(0, 512, total, 768 // 64)  # 12
    assert (w_start, w_end) == (0, 64)
    assert (s_start, s_end) == (0, 8)
    # Both correspond to the same 512/768 fraction of their axis.
    assert w_end / (768 // 8) == s_end / (768 // 64) == 512 / 768


def test_axis_bounds_unaligned_raises():
    # A boundary that doesn't divide the axis evenly must fail loudly.
    with pytest.raises(AssertionError):
        shard_axis_bounds(1, 2, 768, 96)
