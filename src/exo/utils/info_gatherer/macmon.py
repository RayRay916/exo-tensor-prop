import os
from typing import Self

from pydantic import BaseModel

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel


class _TempMetrics(BaseModel, extra="ignore"):
    """Temperature-related metrics returned by macmon."""

    cpu_temp_avg: float
    gpu_temp_avg: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """Memory-related metrics returned by macmon."""

    ram_total: int
    ram_usage: int
    swap_total: int
    swap_usage: int


class RawMacmonMetrics(BaseModel, extra="ignore"):
    """Complete set of metrics returned by macmon.

    Unknown fields are ignored for forward-compatibility.
    """

    timestamp: str  # ignored
    temp: _TempMetrics
    memory: _MemoryMetrics
    ecpu_usage: tuple[int, float]  # freq mhz, usage %
    pcpu_usage: tuple[int, float]  # freq mhz, usage %
    gpu_usage: tuple[int, float]  # freq mhz, usage %
    all_power: float
    ane_power: float
    cpu_power: float
    gpu_power: float
    gpu_ram_power: float
    ram_power: float
    sys_power: float


def _maybe_override_ram_available(default_bytes: int) -> int:
    """If OVERRIDE_MEMORY_MB is set, cap ram_available to that value.

    Mirrors the override that already exists on the psutil fallback path
    (see info_gatherer._monitor_memory_usage), so layer placement weights
    can be biased on macOS without disabling macmon.
    """
    env = os.getenv("OVERRIDE_MEMORY_MB")
    if not env:
        return default_bytes
    try:
        return Memory.from_mb(int(env)).in_bytes
    except (TypeError, ValueError):
        return default_bytes


class MacmonMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile
    memory: MemoryUsage

    @classmethod
    def from_raw(cls, raw: RawMacmonMetrics) -> Self:
        ram_available = _maybe_override_ram_available(
            raw.memory.ram_total - raw.memory.ram_usage
        )
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=raw.gpu_usage[1],
                temp=raw.temp.gpu_temp_avg,
                sys_power=raw.sys_power,
                pcpu_usage=raw.pcpu_usage[1],
                ecpu_usage=raw.ecpu_usage[1],
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.ram_total,
                ram_available=ram_available,
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_usage),
            ),
        )

    @classmethod
    def from_raw_json(cls, json: str) -> Self:
        return cls.from_raw(RawMacmonMetrics.model_validate_json(json))
