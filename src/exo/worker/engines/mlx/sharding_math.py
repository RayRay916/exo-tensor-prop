"""Pure arithmetic for uneven (proportional) tensor sharding.

Kept free of any ``mlx`` import so the boundary math — the part whose subtle
off-by-one or misalignment would silently corrupt weights rather than crash —
can be unit-tested in the fast suite. ``auto_parallel`` wires these into the
actual array slicing.
"""


def aligned_segment_sizes(
    dim: int, proportions: list[float], align: int
) -> list[int]:
    """Split ``dim`` into ``len(proportions)`` integer parts, each a multiple of
    ``align``, approximating ``proportions`` and summing exactly to ``dim``.

    Largest-remainder rounding; every part is forced to at least one
    ``align``-sized unit so no rank gets an empty shard.
    """
    if align < 1:
        align = 1
    assert dim % align == 0, f"dim {dim} is not a multiple of align {align}"
    units = dim // align
    n = len(proportions)
    if units < n:
        raise ValueError(f"cannot split {units} units across {n} ranks")
    raw = [p * units for p in proportions]
    sizes = [int(r) for r in raw]
    leftover = units - sum(sizes)
    for idx in sorted(range(n), key=lambda i: raw[i] - sizes[i], reverse=True):
        if leftover <= 0:
            break
        sizes[idx] += 1
        leftover -= 1
    for i in range(n):
        if sizes[i] == 0:
            donor = max(range(n), key=lambda k: sizes[k])
            assert sizes[donor] > 1
            sizes[donor] -= 1
            sizes[i] = 1
    return [s * align for s in sizes]


def shard_axis_bounds(
    start_unit: int, end_unit: int, total_units: int, dim: int
) -> tuple[int, int]:
    """Map a half-open window of logical units onto an array axis of length
    ``dim`` using **integer** arithmetic.

    A quantized linear stores its packed weight, scales and biases with
    different lengths along the contraction axis (``dim/pack_factor`` vs
    ``dim/group_size``). Computing each array's cut with floats risked rounding
    them onto inconsistent boundaries — silent corruption. Here the boundary is
    ``unit * dim // total_units`` and we assert it divides evenly, so every
    array of a given layer is sliced at the same logical point or we fail loudly.
    """
    assert (start_unit * dim) % total_units == 0 and (
        end_unit * dim
    ) % total_units == 0, (
        f"shard boundary not aligned to axis: dim={dim} "
        f"units=({start_unit},{end_unit})/{total_units}"
    )
    return start_unit * dim // total_units, end_unit * dim // total_units
