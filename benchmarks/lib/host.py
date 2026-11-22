"""Host resource probing, CPU affinity, port selection, math, and DB file utilities."""

from __future__ import annotations

import math
import os
import shutil
import socket
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lib.paths import DATA_DIR


def compute_dynamic_coadd_workers(cpu_cores: int) -> int:
    """Reserve ``max(1, total // 3)`` cores for I/O/asyncio (~2/3 allocated to CPU coadds)."""
    total = max(1, int(cpu_cores))
    reserved = max(1, total // 3)
    return max(1, total - reserved)


def select_physical_first_cpus(vcpus: int) -> list[int]:
    """Select logical CPUs prioritizing distinct physical ``core_id``s before SMT siblings."""
    total_cores = os.cpu_count() or 1
    target = min(max(1, int(vcpus)), total_cores)
    by_core_id: dict[int, list[int]] = defaultdict(list)
    try:
        for cpu_idx in range(total_cores):
            cid_path = Path(f"/sys/devices/system/cpu/cpu{cpu_idx}/topology/core_id")
            if cid_path.exists():
                cid = int(cid_path.read_text(encoding="utf-8").strip())
            else:
                cid = cpu_idx
            by_core_id[cid].append(cpu_idx)
    except OSError:
        return list(range(target))

    ordered: list[int] = []
    max_threads = max((len(v) for v in by_core_id.values()), default=1)
    for t_idx in range(max_threads):
        for cid in sorted(by_core_id):
            cpus = by_core_id[cid]
            if t_idx < len(cpus):
                ordered.append(cpus[t_idx])
    return sorted(ordered[:target])


def apply_cpu_affinity(vcpus: int) -> list[int]:
    """Pin the current process to physical-first CPUs when ``os.sched_setaffinity`` exists."""
    pinned = select_physical_first_cpus(vcpus)
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(pinned))
        except OSError:
            pass
    return pinned


def _read_proc_meminfo() -> dict[str, int]:
    """Read Linux ``/proc/meminfo`` values in MB, returning empty dict if unavailable."""
    mem_info: dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return mem_info
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(":", 1)
            if len(parts) == 2:
                v_parts = parts[1].strip().split()
                if v_parts and v_parts[0].isdigit():
                    mem_info[parts[0].strip()] = int(v_parts[0]) // 1024
    except OSError:
        pass
    return mem_info


def db_group_bytes(db_path: Path) -> int:
    """Return total disk bytes used by ``db_path`` and any SQLite ``-wal`` / ``-shm`` sidecars."""
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            total += p.stat().st_size
    return total


def unlink_db_group(base: Path) -> None:
    """Remove a SQLite database file and its ``-wal`` / ``-shm`` sidecar files if present."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{base}{suffix}")
        if p.exists():
            p.unlink()


def copy_db_group(src: Path, dst: Path) -> None:
    """Copy a SQLite database file and any ``-wal`` / ``-shm`` sidecars cleanly."""
    unlink_db_group(dst)
    for suffix in ("", "-wal", "-shm"):
        sp = Path(f"{src}{suffix}")
        if sp.exists():
            shutil.copy2(sp, Path(f"{dst}{suffix}"))


def probe_host_resources(db_path: Path, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Inspect host RAM, disk headroom, and CPU count to size safe benchmark limits."""
    mem_info = _read_proc_meminfo()
    total_ram_mb = mem_info.get("MemTotal", 4096)
    avail_ram_mb = mem_info.get("MemAvailable", max(1024, total_ram_mb // 2))
    swap_free_mb = mem_info.get("SwapFree", 0)

    stale_db_mb = db_group_bytes(db_path) / (1024 * 1024)
    data_dir.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(data_dir)
    free_disk_now_mb = disk_usage.free / (1024 * 1024)
    effective_free_disk_mb = free_disk_now_mb + stale_db_mb

    ram_headroom_mb = max(256, min(1024, int(total_ram_mb * 0.15)))
    disk_headroom_mb = max(512, min(1536, int(effective_free_disk_mb * 0.10)))

    max_safe_in_memory_mb = max(256, min(4096, int(avail_ram_mb - ram_headroom_mb)))
    max_safe_sqlite_mb = max(512, min(8192, int(effective_free_disk_mb - disk_headroom_mb)))

    cpu_count = os.cpu_count() or 1
    low_vcpus = min(2, cpu_count)
    high_vcpus = min(4, cpu_count)
    low_cw = compute_dynamic_coadd_workers(low_vcpus)
    high_cw = compute_dynamic_coadd_workers(high_vcpus)

    mid_mem = min(512, max_safe_in_memory_mb)
    mid_db = min(1536, max_safe_sqlite_mb)
    high_mem = max_safe_in_memory_mb
    high_db = max_safe_sqlite_mb

    recommended_matrix = [
        {
            "name": f"cfg1_{low_vcpus}vcpu_48m_ram",
            "fly_equivalent": (
                f"shared-cpu-{low_vcpus}x (Constrained Cache: 48MB LRU / 200MB SQLite, "
                f"{low_cw} coadd / {max(1, low_vcpus - low_cw)} asyncio)"
            ),
            "vcpus": low_vcpus,
            "coadd_workers": low_cw,
            "pinned_cpus": select_physical_first_cpus(low_vcpus),
            "in_memory_cache_mb": 48,
            "sqlite_max_cache_mb": 200,
        },
        {
            "name": f"cfg2_{high_vcpus}vcpu_{mid_mem}m_ram",
            "fly_equivalent": (
                f"shared-cpu-{high_vcpus}x ({mid_mem}MB LRU / {mid_db}MB SQLite, "
                f"{high_cw} coadd / {max(1, high_vcpus - high_cw)} asyncio)"
            ),
            "vcpus": high_vcpus,
            "coadd_workers": high_cw,
            "pinned_cpus": select_physical_first_cpus(high_vcpus),
            "in_memory_cache_mb": mid_mem,
            "sqlite_max_cache_mb": mid_db,
        },
    ]
    if high_mem > mid_mem + 128 or high_db > mid_db + 256:
        recommended_matrix.append(
            {
                "name": f"cfg3_{high_vcpus}vcpu_{high_mem}m_max_host",
                "fly_equivalent": (
                    f"shared-cpu-{high_vcpus}x ({high_mem}MB LRU / {high_db}MB SQLite, "
                    f"{high_cw} coadd / {max(1, high_vcpus - high_cw)} asyncio)"
                ),
                "vcpus": high_vcpus,
                "coadd_workers": high_cw,
                "pinned_cpus": select_physical_first_cpus(high_vcpus),
                "in_memory_cache_mb": high_mem,
                "sqlite_max_cache_mb": high_db,
            }
        )

    return {
        "cpu_cores_available": cpu_count,
        "total_ram_mb": total_ram_mb,
        "avail_ram_mb": avail_ram_mb,
        "swap_free_mb": swap_free_mb,
        "free_disk_now_mb": round(free_disk_now_mb, 1),
        "stale_bench_db_mb": round(stale_db_mb, 1),
        "effective_free_disk_mb": round(effective_free_disk_mb, 1),
        "max_safe_in_memory_mb": max_safe_in_memory_mb,
        "max_safe_sqlite_mb": max_safe_sqlite_mb,
        "recommended_matrix": recommended_matrix,
    }


def pick_free_port(preferred: int = 18080) -> int:
    """Return ``preferred`` if bindable on localhost, otherwise an ephemeral free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def percentiles(values: Sequence[float], pcts: Sequence[float]) -> tuple[float, ...]:
    """Compute multiple linear-interpolated percentiles using a single sort pass."""
    if not values:
        return tuple(0.0 for _ in pcts)
    ordered = sorted(values)
    last_idx = len(ordered) - 1
    out: list[float] = []
    for pct in pcts:
        idx = last_idx * (pct / 100.0)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            out.append(float(ordered[lo]))
        else:
            frac = idx - lo
            out.append(float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac))
    return tuple(out)
