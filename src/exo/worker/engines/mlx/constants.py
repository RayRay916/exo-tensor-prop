import os
import warnings

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?


def _parse_kv_cache_bits() -> int | None:
    """Opt-in KV-cache quantization via EXO_KV_BITS; unset => off.

    Validate against the bit widths MLX's QuantizedKVCache supports so a typo
    can't crash the runner at import with an opaque traceback — fall back to
    off (None) with a warning instead.
    """
    raw = os.environ.get("EXO_KV_BITS")
    if not raw:
        return None
    supported = {2, 4, 8}
    try:
        bits = int(raw)
    except ValueError:
        warnings.warn(
            f"EXO_KV_BITS={raw!r} is not an integer; KV-cache quant off", stacklevel=2
        )
        return None
    if bits not in supported:
        warnings.warn(
            f"EXO_KV_BITS={bits} unsupported (expected one of {sorted(supported)}); "
            "KV-cache quant off",
            stacklevel=2,
        )
        return None
    return bits

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = _parse_kv_cache_bits()

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
