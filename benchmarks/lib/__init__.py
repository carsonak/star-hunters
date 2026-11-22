"""Shared modular helpers for the Star Hunters benchmark and stress-test suite."""

from __future__ import annotations

from lib.host import (
    apply_cpu_affinity,
    compute_dynamic_coadd_workers,
    copy_db_group,
    percentiles,
    pick_free_port,
    probe_host_resources,
    select_physical_first_cpus,
    unlink_db_group,
)
from lib.http_load import fetch_patch_and_frames, summarize_endpoints, timed_get
from lib.instrumentation import (
    instrument_cache,
    register_bench_stats_route,
    seed_benchmark_ephemeris,
)
from lib.paths import (
    BENCH_DIR,
    DATA_DIR,
    DEFAULT_BACKEND_DIR,
    RESULTS_DIR,
    ensure_runtime_dirs,
    resolve_backend_dir,
)
from lib.telemetry import ProcTelemetrySampler

__all__ = [
    "BENCH_DIR",
    "DATA_DIR",
    "DEFAULT_BACKEND_DIR",
    "RESULTS_DIR",
    "ProcTelemetrySampler",
    "apply_cpu_affinity",
    "compute_dynamic_coadd_workers",
    "copy_db_group",
    "ensure_runtime_dirs",
    "fetch_patch_and_frames",
    "instrument_cache",
    "percentiles",
    "pick_free_port",
    "probe_host_resources",
    "register_bench_stats_route",
    "resolve_backend_dir",
    "seed_benchmark_ephemeris",
    "select_physical_first_cpus",
    "summarize_endpoints",
    "timed_get",
    "unlink_db_group",
]
