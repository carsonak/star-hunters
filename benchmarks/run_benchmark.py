#!/usr/bin/env python3
"""Multi-configuration two-phase 20 connections/minute full-stack backend benchmark.

Key capabilities:
- 3-minute arrival windows (60 connections @ 20 conn/min over 180s per phase).
- High-variance user staying time tightened to 60s-140s across 3 explorer personas
  (Quick Explorer ~60-80s, Standard Researcher ~85-110s, Deep Astrometry Hunter ~115-140s).
- Hard concurrency ceiling of 25 active virtual users (configurable via --max-concurrent).
- Dynamic archival coadd concurrency formula:
  <= 6 cores -> reserve half for asyncio; > 6 cores -> reserve a quarter for asyncio.
- Physical-core-first Linux CPU affinity (os.sched_setaffinity) so 2-vCPU tests pick
  distinct physical cores (e.g. {0, 2}) before SMT sibling threads ({1, 3}).
- Request-weighted mean / p50 / p95 / p99 latencies across all endpoints and heavy endpoints.
- All outputs stored in /home/line/hackathons/star-hunters-workspace/benchmarks/ (outside git).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent / "backend"
CITIES_CSV = BACKEND_DIR / "app" / "models" / "world_capital_cities.csv"
DB_PATH = BENCH_DIR / "bench_cache.db"
RESULTS_PATH = BENCH_DIR / "benchmark_results.json"

HOT_EXTRA_COORDS: list[tuple[str, float, float]] = [
    ("andromeda_core", 10.6847, 41.2687),
    ("orion_trapezium", 83.8221, -5.3911),
    ("m87_core", 187.7059, 12.3911),
    ("omega_centauri", 201.6970, -47.4795),
    ("galactic_center", 266.4168, -29.0078),
    ("vega_field", 279.2347, 38.7837),
    ("cygnus_x", 310.3580, 45.2800),
    ("pegasus_deep", 344.4127, 20.7688),
]

SEARCH_QUERIES = [
    "mars",
    "saturn",
    "barnard",
    "luhman",
    "ceres",
    "atlas",
    "wise",
    "vega",
    "proxima",
    "trappist",
]

MOST_TARGETS = ["Ceres", "Pallas", "Vesta", "Juno"]


def compute_dynamic_coadd_workers(cpu_cores: int) -> int:
    """If <= 6 cores reserve half for IO/asyncio; if > 6 cores reserve max(3, ceil(total/4))."""
    total = max(1, int(cpu_cores))
    if total <= 6:
        reserved = max(1, math.ceil(total / 2))
    else:
        reserved = max(3, math.ceil(total / 4))
    return max(1, total - reserved)


def select_physical_first_cpus(vcpus: int) -> list[int]:
    """Select logical CPUs prioritizing distinct physical core_ids before SMT siblings."""
    total_cores = os.cpu_count() or 4
    target = min(max(1, vcpus), total_cores)
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


def probe_wsl_resources(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Inspect current WSL /proc/meminfo and filesystem free space for safe sizing."""
    mem_info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) == 2:
                k = parts[0].strip()
                v_parts = parts[1].strip().split()
                if v_parts and v_parts[0].isdigit():
                    mem_info[k] = int(v_parts[0]) // 1024  # MB

    stale_db_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            stale_db_bytes += p.stat().st_size
    stale_db_mb = stale_db_bytes / (1024 * 1024)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(BENCH_DIR)
    free_disk_now_mb = (st.f_bavail * st.f_frsize) / (1024 * 1024)
    effective_free_disk_mb = free_disk_now_mb + stale_db_mb

    avail_ram_mb = mem_info.get("MemAvailable", 1024)
    total_ram_mb = mem_info.get("MemTotal", 4096)
    swap_free_mb = mem_info.get("SwapFree", 0)

    max_safe_in_memory_mb = max(256, min(3072, int(avail_ram_mb - 900)))
    max_safe_sqlite_mb = max(750, min(6144, int(effective_free_disk_mb - 900)))

    cpu_count = os.cpu_count() or 4

    # With the 4.2x RAM cache fix (~0.95 MB/patch in RAM, ~2.58 MB/patch in SQLite),
    # 232 total patches take ~220 MB in RAM (~72 MB for the 76 reusable hot/city patches)
    # and ~600 MB in SQLite (~195 MB for the 76 reusable patches).
    # cfg1 tests eviction-constrained caches (48 MB RAM / 200 MB SQLite) on 2 vCPUs;
    # cfg2 tests balanced Fly.io caps (512 MB RAM / 1536 MB SQLite) on 4 vCPUs;
    # cfg3 tests high/max WSL caps (1024+ MB RAM / 3072+ MB SQLite) on 4 vCPUs.
    low_vcpus = min(2, cpu_count)
    high_vcpus = min(4, cpu_count)
    mid_mem = min(512, max_safe_in_memory_mb)
    mid_db = min(1536, max_safe_sqlite_mb)
    high_mem = max_safe_in_memory_mb
    high_db = max_safe_sqlite_mb

    recommended_matrix = [
        {
            "name": "cfg1_2vcpu_48m_ram",
            "fly_equivalent": "shared-cpu-2x (Constrained Cache: 48MB LRU / 200MB SQLite, 1 coadd / 1 asyncio)",
            "vcpus": low_vcpus,
            "coadd_workers": compute_dynamic_coadd_workers(low_vcpus),
            "pinned_cpus": select_physical_first_cpus(low_vcpus),
            "in_memory_cache_mb": 48,
            "sqlite_max_cache_mb": 200,
        },
        {
            "name": f"cfg2_4vcpu_{mid_mem}m_ram",
            "fly_equivalent": f"shared-cpu-4x (4096MB RAM, {mid_mem}MB LRU / {mid_db}MB SQLite, 2 coadd / 2 asyncio)",
            "vcpus": high_vcpus,
            "coadd_workers": compute_dynamic_coadd_workers(high_vcpus),
            "pinned_cpus": select_physical_first_cpus(high_vcpus),
            "in_memory_cache_mb": mid_mem,
            "sqlite_max_cache_mb": mid_db,
        },
    ]
    if high_mem > mid_mem + 128 or high_db > mid_db + 256:
        recommended_matrix.append(
            {
                "name": f"cfg3_4vcpu_{high_mem}m_max_wsl",
                "fly_equivalent": f"shared-cpu-4x (4096-8192MB RAM, {high_mem}MB LRU / {high_db}MB SQLite, 2 coadd / 2 asyncio)",
                "vcpus": high_vcpus,
                "coadd_workers": compute_dynamic_coadd_workers(high_vcpus),
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


def _pick_free_port(preferred: int = 18080) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def _angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(a1 - a2)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def _load_60_cities() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CITIES_CSV.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "country": row["Country"].strip(),
                    "city": row["Capital City"].strip(),
                    "lat": lat,
                    "lon": lon,
                }
            )
    rows.sort(key=lambda r: (r["city"], r["country"]))
    step = max(1, len(rows) // 60)
    sampled = rows[::step][:60]
    if len(sampled) < 60:
        seen = {(r["city"], r["country"]) for r in sampled}
        for r in rows:
            if (r["city"], r["country"]) not in seen:
                sampled.append(r)
                seen.add((r["city"], r["country"]))
            if len(sampled) == 60:
                break
    return sampled


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


# ---------------------------------------------------------------------------
# Dedicated Benchmarked FastAPI Server Subprocess (--serve)
# ---------------------------------------------------------------------------


def run_bench_server(
    port: int,
    db_path: Path,
    vcpus: int,
    coadd_workers: int,
    in_memory_mb: int,
    sqlite_max_mb: int,
) -> None:
    pinned_cpus = select_physical_first_cpus(vcpus)
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(pinned_cpus))
        except OSError:
            pass

    resolved_coadd_workers = (
        coadd_workers if coadd_workers > 0 else compute_dynamic_coadd_workers(vcpus)
    )

    sys.path.insert(0, str(BACKEND_DIR))
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    os.environ["ENABLE_ARCHIVAL_DATA"] = "true"
    os.environ["IN_MEMORY_CACHE_MB"] = str(in_memory_mb)
    os.environ["ARCHIVAL_MAX_CONCURRENT_COADDS"] = str(resolved_coadd_workers)
    os.environ["COLD_MISS_RATE_LIMIT_PER_IP_PER_MIN"] = "0"
    os.environ["LOG_LEVEL"] = "WARNING"

    import uvicorn
    from app.main import create_app
    from app.services import backend_service as bs_mod
    from app.services import catalog_clients as cc_mod
    from app.services import celestial_catalog_pipeline as ccp_mod
    from app.services import object_epochs as oe_mod
    from app.services.backend_service import StarHuntersBackendService
    from app.services.spatial_cache import SpatialSqliteCache
    from app.services.spectrum_extractor import extract_forced_spectrum
    from app.services.spherex_synthetic import generate_synthetic_spherex_cutout
    from app.utils.astronomy import datetime_to_mjd

    cache = SpatialSqliteCache(
        db_path=str(db_path),
        max_cache_bytes=int(sqlite_max_mb) * 1024 * 1024,
        max_memory_bytes=int(in_memory_mb) * 1024 * 1024,
    )

    now_mjd = datetime_to_mjd()
    ephem_points = []
    for idx, spec in enumerate(cc_mod.HORIZONS_FEATURED_BODIES):
        base_ra = (idx * 37.5 + 15.0) % 360.0
        base_dec = -20.0 + (idx % 7) * 6.5
        for delta_day in (-1.0, 0.0, 1.0):
            ephem_points.append(
                {
                    "naif_id": spec["naif_id"],
                    "body_name": spec["name"],
                    "body_class": spec["body_class"],
                    "mjd": now_mjd + delta_day,
                    "ra": (base_ra + delta_day * 0.25) % 360.0,
                    "dec": base_dec + delta_day * 0.08,
                    "vmag": 5.5 + (idx % 6) * 1.1,
                    "source": "jpl_horizons_bench",
                }
            )
    cache.upsert_ephemeris_points(ephem_points)

    stats_lock = threading.Lock()
    stats: dict[str, Any] = {
        "mem_lru_raw_hits": 0,
        "mem_lru_raw_misses": 0,
        "tier_outcomes": defaultdict(int),
        "cold_writes": defaultdict(int),
        "cold_coadd_calls": 0,
        "cold_spectrum_calls": 0,
    }

    orig_mem_get = cache._mem_cache.get

    def tracked_mem_get(key: str) -> bytes | None:
        res = orig_mem_get(key)
        with stats_lock:
            if res is not None:
                stats["mem_lru_raw_hits"] += 1
            else:
                stats["mem_lru_raw_misses"] += 1
        return res

    cache._mem_cache.get = tracked_mem_get  # type: ignore[method-assign]

    def _in_mem_peek(key: str) -> bool:
        with cache._mem_cache._lock:
            return key in cache._mem_cache._entries

    def _record_tier(kind: str, in_ram: bool, hit: bool) -> None:
        with stats_lock:
            if not hit:
                stats["tier_outcomes"][f"{kind}:miss"] += 1
            elif in_ram:
                stats["tier_outcomes"][f"{kind}:in_memory_lru"] += 1
            else:
                stats["tier_outcomes"][f"{kind}:sqlite_disk"] += 1

    orig_get_manifest = cache.get_manifest
    orig_get_catalogs = cache.get_catalogs
    orig_get_frame_gzip = cache.get_frame_gzip
    orig_get_frame = cache.get_frame
    orig_get_spectrum = cache.get_spectrum
    orig_get_zenith = cache.get_zenith_response
    orig_put_patch = cache.put_sky_patch
    orig_put_spec = cache.put_spectrum
    orig_put_zenith = cache.put_zenith_response

    @functools.wraps(orig_get_manifest)
    def wrap_get_manifest(grid_key: str) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"manifest:{grid_key}")
        res = orig_get_manifest(grid_key)
        _record_tier("manifest", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_catalogs)
    def wrap_get_catalogs(grid_key: str) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"catalogs:{grid_key}")
        res = orig_get_catalogs(grid_key)
        _record_tier("catalogs", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_frame_gzip)
    def wrap_get_frame_gzip(grid_key: str, filename: str, precision: str = "float32") -> Any:
        dtype_hdr = "uint8" if filename.endswith("_flags.bin") else precision
        in_ram = _in_mem_peek(f"frame_gz:{grid_key}:{filename}:{dtype_hdr}") or _in_mem_peek(
            f"frame_raw:{grid_key}:{filename}"
        )
        res = orig_get_frame_gzip(grid_key, filename, precision=precision)  # type: ignore[arg-type]
        _record_tier("frame_gzip", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_frame)
    def wrap_get_frame(grid_key: str, filename: str) -> bytes | None:
        in_ram = _in_mem_peek(f"frame_raw:{grid_key}:{filename}")
        res = orig_get_frame(grid_key, filename)
        _record_tier("frame_raw", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_spectrum)
    def wrap_get_spectrum(spectrum_key: str) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"spectrum:{spectrum_key}")
        res = orig_get_spectrum(spectrum_key)
        _record_tier("spectrum", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_zenith)
    def wrap_get_zenith(cache_key: str) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"zenith:{cache_key}")
        res = orig_get_zenith(cache_key)
        _record_tier("zenith", in_ram, res is not None)
        return res

    @functools.wraps(orig_put_patch)
    def wrap_put_patch(*args: Any, **kwargs: Any) -> None:
        orig_put_patch(*args, **kwargs)
        with stats_lock:
            stats["cold_writes"]["sky_patch"] += 1

    @functools.wraps(orig_put_spec)
    def wrap_put_spec(*args: Any, **kwargs: Any) -> None:
        orig_put_spec(*args, **kwargs)
        with stats_lock:
            stats["cold_writes"]["spectrum"] += 1

    @functools.wraps(orig_put_zenith)
    def wrap_put_zenith(*args: Any, **kwargs: Any) -> None:
        orig_put_zenith(*args, **kwargs)
        with stats_lock:
            stats["cold_writes"]["zenith"] += 1

    orig_peek_manifest = cache.peek_manifest_memory
    orig_peek_catalogs = cache.peek_catalogs_memory
    orig_peek_spectrum = cache.peek_spectrum_memory
    orig_peek_frame_gzip = cache.peek_frame_gzip_memory

    @functools.wraps(orig_peek_manifest)
    def wrap_peek_manifest(grid_key: str) -> dict[str, Any] | None:
        res = orig_peek_manifest(grid_key)
        if res is not None:
            _record_tier("manifest", True, True)
        return res

    @functools.wraps(orig_peek_catalogs)
    def wrap_peek_catalogs(grid_key: str) -> dict[str, Any] | None:
        res = orig_peek_catalogs(grid_key)
        if res is not None:
            _record_tier("catalogs", True, True)
        return res

    @functools.wraps(orig_peek_spectrum)
    def wrap_peek_spectrum(spectrum_key: str) -> dict[str, Any] | None:
        res = orig_peek_spectrum(spectrum_key)
        if res is not None:
            _record_tier("spectrum", True, True)
        return res

    @functools.wraps(orig_peek_frame_gzip)
    def wrap_peek_frame_gzip(grid_key: str, filename: str, precision: str = "float32") -> Any:
        res = orig_peek_frame_gzip(grid_key, filename, precision=precision)  # type: ignore[arg-type]
        if res is not None:
            _record_tier("frame_gzip", True, True)
        return res

    cache.get_manifest = wrap_get_manifest  # type: ignore[method-assign]
    cache.get_catalogs = wrap_get_catalogs  # type: ignore[method-assign]
    cache.get_frame_gzip = wrap_get_frame_gzip  # type: ignore[method-assign]
    cache.get_frame = wrap_get_frame  # type: ignore[method-assign]
    cache.get_spectrum = wrap_get_spectrum  # type: ignore[method-assign]
    cache.get_zenith_response = wrap_get_zenith  # type: ignore[method-assign]
    cache.peek_manifest_memory = wrap_peek_manifest  # type: ignore[method-assign]
    cache.peek_catalogs_memory = wrap_peek_catalogs  # type: ignore[method-assign]
    cache.peek_spectrum_memory = wrap_peek_spectrum  # type: ignore[method-assign]
    cache.peek_frame_gzip_memory = wrap_peek_frame_gzip  # type: ignore[method-assign]
    cache.put_sky_patch = wrap_put_patch  # type: ignore[method-assign]
    cache.put_spectrum = wrap_put_spec  # type: ignore[method-assign]
    cache.put_zenith_response = wrap_put_zenith  # type: ignore[method-assign]

    def mock_cutout_fetcher(
        ra: float,
        dec: float,
        fov_arcmin: float = 5.0,
        max_cutouts: int | None = None,
        **_kwargs: Any,
    ) -> list[Any]:
        time.sleep(0.03)
        mjds = (60500.0,) if (max_cutouts is not None and max_cutouts <= 6) else (60500.0, 60682.5)
        cutouts = []
        for mjd in mjds:
            for det in range(1, 7):
                cutouts.append(
                    generate_synthetic_spherex_cutout(
                        ra=ra,
                        dec=dec,
                        fov_arcmin=fov_arcmin,
                        bandpass=f"SPHEREx-D{det}",
                        shape=(64, 64),
                        mjd_obs=mjd,
                    )
                )
        with stats_lock:
            if len(mjds) == 2:
                stats["cold_coadd_calls"] += 1
            else:
                stats["cold_spectrum_calls"] += 1
        return cutouts

    def disabled_fallback(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("External archive fallback disabled in local benchmark")

    async def no_live_fetch(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        return [], "bench_offline"

    async def no_live_horizons(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def no_live_most(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"query_name": kwargs.get("obj_name", ""), "status": "no_match", "matches": []}

    # Prevent all outbound HTTP calls to external astronomy servers (including object_epochs SIA)
    ccp_mod.fetch_simbad_objects = no_live_fetch  # type: ignore[assignment]
    ccp_mod.fetch_gaia_dr3_objects = no_live_fetch  # type: ignore[assignment]
    bs_mod.fetch_most_spherex_matches = no_live_most  # type: ignore[assignment]
    oe_mod.query_observations = lambda *_a, **_kw: []  # type: ignore[assignment]

    service = StarHuntersBackendService(
        cache=cache,
        enable_archival_data=True,
        wise_fallback_fetcher=disabled_fallback,
        skyview_fallback_fetcher=disabled_fallback,
        spherex_cutout_fetcher=mock_cutout_fetcher,
        spectrum_extractor=extract_forced_spectrum,
    )
    service.catalog_pipeline.resolve_live_horizons_for_body = no_live_horizons  # type: ignore[method-assign]

    app = create_app(service)

    @app.get("/internal/bench-cache-stats")
    async def bench_cache_stats() -> dict[str, Any]:
        with stats_lock:
            tier_copy = dict(stats["tier_outcomes"])
            writes_copy = dict(stats["cold_writes"])
            raw_hits = int(stats["mem_lru_raw_hits"])
            raw_misses = int(stats["mem_lru_raw_misses"])
            coadd_calls = int(stats["cold_coadd_calls"])
            spec_calls = int(stats["cold_spectrum_calls"])

        with cache.connection() as conn:
            patch_row = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(byte_size), 0) AS bytes FROM sky_patches"
            ).fetchone()
            frame_row = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(LENGTH(compressed_blob)), 0) AS comp_bytes, "
                "COALESCE(SUM(raw_byte_size), 0) AS raw_bytes FROM binary_frames"
            ).fetchone()
            spec_row = conn.execute("SELECT COUNT(*) AS cnt FROM spectra_cache").fetchone()
            zen_row = conn.execute("SELECT COUNT(*) AS cnt FROM capital_zenith_cache").fetchone()

        return {
            "mem_lru_raw_hits": raw_hits,
            "mem_lru_raw_misses": raw_misses,
            "mem_lru_current_bytes": cache._mem_cache.current_bytes,
            "mem_lru_max_bytes": cache._mem_cache.max_bytes,
            "mem_lru_entries": cache._mem_cache.entry_count,
            "sqlite_max_bytes": cache.max_cache_bytes,
            "tier_outcomes": tier_copy,
            "cold_writes": writes_copy,
            "cold_coadd_calls": coadd_calls,
            "cold_spectrum_calls": spec_calls,
            "sqlite_tables": {
                "sky_patches_count": int(patch_row["cnt"]),
                "sky_patches_bytes": int(patch_row["bytes"]),
                "binary_frames_count": int(frame_row["cnt"]),
                "binary_frames_compressed_bytes": int(frame_row["comp_bytes"]),
                "binary_frames_raw_bytes": int(frame_row["raw_bytes"]),
                "spectra_count": int(spec_row["cnt"]),
                "zenith_count": int(zen_row["cnt"]),
            },
        }

    bench_route = app.router.routes.pop()
    app.router.routes.insert(0, bench_route)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------------------
# /proc/<pid> Kernel Telemetry Sampler
# ---------------------------------------------------------------------------


class ProcTelemetrySampler:
    """Samples CPU %, RSS/HWM RAM, Kernel Disk I/O, and SQLite disk sizes every 250ms."""

    def __init__(self, pid: int, db_path: Path, base_url: str) -> None:
        self.pid = pid
        self.db_path = db_path
        self.base_url = base_url
        self.clk_tck = os.sysconf("SC_CLK_TCK")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _read_proc_stat(self) -> float:
        with open(f"/proc/{self.pid}/stat", encoding="utf-8") as fh:
            raw = fh.read()
        rparen = raw.rfind(")")
        fields = raw[rparen + 2 :].split()
        utime = int(fields[11])
        stime = int(fields[12])
        return (utime + stime) / self.clk_tck

    def _read_proc_status(self) -> tuple[float, float, int]:
        rss_kb = 0.0
        hwm_kb = 0.0
        threads = 1
        with open(f"/proc/{self.pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = float(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
        return rss_kb / 1024.0, hwm_kb / 1024.0, threads

    def _read_proc_io(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            with open(f"/proc/{self.pid}/io", encoding="utf-8") as fh:
                for line in fh:
                    k, v = line.split(":")
                    out[k.strip()] = int(v.strip())
        except OSError:
            pass
        return out

    def _db_disk_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{self.db_path}{suffix}")
            if p.exists():
                total += p.stat().st_size
        return total

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            return list(self.samples)

    def reset_samples(self) -> None:
        with self._lock:
            self.samples.clear()

    def _run(self) -> None:
        prev_cpu = self._read_proc_stat()
        prev_wall = time.perf_counter()
        while not self._stop.wait(0.25):
            try:
                now_wall = time.perf_counter()
                now_cpu = self._read_proc_stat()
                dt_wall = max(1e-6, now_wall - prev_wall)
                cpu_pct = ((now_cpu - prev_cpu) / dt_wall) * 100.0
                prev_cpu = now_cpu
                prev_wall = now_wall

                rss_mb, hwm_mb, threads = self._read_proc_status()
                io_stats = self._read_proc_io()
                db_mb = self._db_disk_bytes() / (1024.0 * 1024.0)

                sample = {
                    "t": round(now_wall, 3),
                    "cpu_pct_total": round(cpu_pct, 2),
                    "rss_mb": round(rss_mb, 2),
                    "hwm_mb": round(hwm_mb, 2),
                    "threads": threads,
                    "io_read_bytes": io_stats.get("read_bytes", 0),
                    "io_write_bytes": io_stats.get("write_bytes", 0),
                    "io_rchar": io_stats.get("rchar", 0),
                    "io_wchar": io_stats.get("wchar", 0),
                    "db_disk_mb": round(db_mb, 3),
                }
                with self._lock:
                    self.samples.append(sample)
            except OSError:
                break


# ---------------------------------------------------------------------------
# Async 20-Connections/Min (3-Min, 60-140s Stay Time, Max 25 Concurrent) Load Generator
# ---------------------------------------------------------------------------


def _generate_cold_coords(
    seed: int,
    count: int,
    forbidden: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    rng = random.Random(seed + 999)
    chosen: list[tuple[float, float]] = []
    attempts = 0
    while len(chosen) < count and attempts < 80_000:
        attempts += 1
        ra = round(rng.uniform(0.0, 359.8), 4)
        dec = round(rng.uniform(-75.0, 75.0), 4)
        if any(_angular_sep_deg(ra, dec, fra, fdec) < 2.2 for fra, fdec in forbidden):
            continue
        if any(_angular_sep_deg(ra, dec, cra, cdec) < 2.2 for cra, cdec in chosen):
            continue
        chosen.append((ra, dec))
    return chosen


def _build_arrival_offsets(
    rng: random.Random,
    total_sessions: int = 60,
    duration_s: float = 180.0,
) -> list[float]:
    """Schedule connections in bursty waves at 20 conn/min over duration_s."""
    num_waves = max(1, total_sessions // 3)
    wave_step = duration_s / num_waves
    offsets: list[float] = []
    for w in range(num_waves):
        center = 0.2 if w == 0 else w * wave_step
        for idx in range(3):
            jitter = rng.uniform(0.0, 1.5) if w == 0 else rng.uniform(-1.8, 2.2)
            t = max(0.0, min(duration_s - 2.0, center + jitter + idx * 0.35))
            offsets.append(t)
    while len(offsets) < total_sessions:
        offsets.append(rng.uniform(0.0, max(1.0, duration_s - 2.0)))
    offsets.sort()
    return offsets[:total_sessions]


def _pick_persona(user_rng: random.Random, smoke_test: bool = False) -> dict[str, Any]:
    """Assign a staying-time persona spanning 60s to 140s with corresponding activity counts."""
    if smoke_test:
        return {
            "name": "smoke",
            "target_stay_s": user_rng.uniform(1.5, 2.5),
            "fov_sweeps": 1,
            "searches": 1,
            "check_object_epochs": False,
            "check_most": False,
            "cold_patches": 1,
            "hot_patches": 1,
            "city_zenith_patches": 1,
            "spectra_prob": 0.5,
        }
    roll = user_rng.random()
    if roll < 0.25:
        return {
            "name": "quick_explorer",
            "target_stay_s": user_rng.uniform(60.0, 80.0),
            "fov_sweeps": 3,
            "searches": 2,
            "check_object_epochs": False,
            "check_most": False,
            "cold_patches": 1,
            "hot_patches": 3,
            "city_zenith_patches": 1,
            "spectra_prob": 0.45,
        }
    if roll < 0.75:
        return {
            "name": "standard_researcher",
            "target_stay_s": user_rng.uniform(85.0, 110.0),
            "fov_sweeps": 5,
            "searches": 3,
            "check_object_epochs": True,
            "check_most": False,
            "cold_patches": 1,
            "hot_patches": 4,
            "city_zenith_patches": 2,
            "spectra_prob": 0.55,
        }
    return {
        "name": "deep_hunter",
        "target_stay_s": user_rng.uniform(115.0, 140.0),
        "fov_sweeps": 6,
        "searches": 3,
        "check_object_epochs": True,
        "check_most": True,
        "cold_patches": 2,
        "hot_patches": 5,
        "city_zenith_patches": 2,
        "spectra_prob": 0.65,
    }


async def _timed_get(
    client: Any,
    url: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    endpoint: str,
    *,
    track_cached: bool = False,
) -> Any | None:
    import httpx

    t0 = time.perf_counter()
    try:
        resp = await client.get(url, params=params)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rec: dict[str, Any] = {
            "endpoint": endpoint,
            "status": resp.status_code,
            "ms": elapsed_ms,
            "wire_bytes": resp.num_bytes_downloaded,
            "raw_bytes": len(resp.content),
        }
        if track_cached:
            rec["cached"] = (
                resp.json().get("query", {}).get("cached") if resp.status_code == 200 else None
            )
        records.append(rec)
        return resp
    except httpx.TimeoutException:
        records.append(
            {
                "endpoint": endpoint,
                "status": 599,
                "ms": (time.perf_counter() - t0) * 1000.0,
                "wire_bytes": 0,
                "raw_bytes": 0,
            }
        )
        return None
    except httpx.HTTPError:
        records.append(
            {
                "endpoint": endpoint,
                "status": 598,
                "ms": (time.perf_counter() - t0) * 1000.0,
                "wire_bytes": 0,
                "raw_bytes": 0,
            }
        )
        return None


async def _fetch_patch_and_frames(
    client: Any,
    ra: float,
    dec: float,
    records: list[dict[str, Any]],
    endpoint_tag: str,
    fetch_spectrum: bool = False,
) -> None:
    await _timed_get(
        client,
        "/api/v1/catalogs",
        {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
        records,
        f"catalogs:{endpoint_tag}",
    )

    patch_resp = await _timed_get(
        client,
        "/api/v1/sky-patch",
        {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
        records,
        f"sky_patch:{endpoint_tag}",
        track_cached=True,
    )
    if patch_resp is None or patch_resp.status_code != 200:
        return
    manifest = patch_resp.json()

    frame_urls: list[tuple[str, dict[str, str]]] = []
    for ep in manifest.get("epochs", []):
        det_url = ep.get("detectors_data_url")
        flg_url = ep.get("flags_mask_url")
        if det_url:
            frame_urls.append((det_url, {"precision": "float16"}))
        if flg_url:
            frame_urls.append((flg_url, {}))

    await asyncio.gather(
        *(
            _timed_get(client, u, p, records, f"frame:{endpoint_tag}")
            for u, p in frame_urls
        )
    )

    if fetch_spectrum:
        await _timed_get(
            client,
            "/api/v1/spectrum",
            {"ra": ra, "dec": dec, "label": f"Target {ra:.2f},{dec:.2f}"},
            records,
            f"spectrum:{endpoint_tag}",
        )


def _compute_weighted_latency_summary(
    endpoint_summary: dict[str, Any],
    *,
    prefixes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    selected = [
        v
        for k, v in endpoint_summary.items()
        if prefixes is None or k.startswith(prefixes)
    ]
    tot_count = sum(v["count"] for v in selected)
    if tot_count <= 0:
        return {"mean_ms": 0.0, "weighted_p50_ms": 0.0, "weighted_p95_ms": 0.0, "weighted_p99_ms": 0.0}
    return {
        "mean_ms": round(sum(v["mean_ms"] * v["count"] for v in selected) / tot_count, 2),
        "weighted_p50_ms": round(sum(v["p50_ms"] * v["count"] for v in selected) / tot_count, 2),
        "weighted_p95_ms": round(sum(v["p95_ms"] * v["count"] for v in selected) / tot_count, 2),
        "weighted_p99_ms": round(sum(v["p99_ms"] * v["count"] for v in selected) / tot_count, 2),
    }


async def run_phase(
    phase_name: str,
    seed: int,
    base_url: str,
    hot_pool: list[tuple[str, float, float]],
    cities_60: list[dict[str, Any]],
    cold_coords: list[tuple[float, float]],
    sampler: ProcTelemetrySampler,
    *,
    total_sessions: int = 60,
    duration_s: float = 180.0,
    max_concurrent_users: int = 25,
    vcpus: int = 4,
    smoke_test: bool = False,
) -> dict[str, Any]:
    import httpx

    rng = random.Random(seed)
    arrival_offsets = _build_arrival_offsets(rng, total_sessions=total_sessions, duration_s=duration_s)
    records: list[dict[str, Any]] = []
    stay_times_s: list[float] = []
    persona_counts: dict[str, int] = defaultdict(int)
    active_sessions = 0
    max_concurrent = 0
    concurrency_samples: list[int] = []
    concurrency_sem = asyncio.Semaphore(max_concurrent_users)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept-Encoding": "gzip"},
        limits=httpx.Limits(max_connections=80, max_keepalive_connections=40),
    ) as client:
        before_resp = await client.get("/internal/bench-cache-stats")
        before_resp.raise_for_status()
        before_cache = before_resp.json()
        sampler.reset_samples()
        phase_t0 = time.perf_counter()

        async def _virtual_user(user_idx: int, start_delay: float) -> None:
            nonlocal active_sessions, max_concurrent
            await asyncio.sleep(start_delay)
            async with concurrency_sem:
                user_t0 = time.perf_counter()
                user_rng = random.Random(seed * 1000 + user_idx)
                persona = _pick_persona(user_rng, smoke_test=smoke_test)
                persona_counts[persona["name"]] += 1
                target_stay_s = float(persona["target_stay_s"])

                total_pause_slots = 6
                slot_target = target_stay_s / total_pause_slots

                async def _pace_until(slot_index: int) -> None:
                    desired_elapsed = slot_index * slot_target
                    actual_elapsed = time.perf_counter() - user_t0
                    remaining = desired_elapsed - actual_elapsed
                    if remaining > 0.02:
                        await asyncio.sleep(remaining)

                active_sessions += 1
                max_concurrent = max(max_concurrent, active_sessions)
                concurrency_samples.append(active_sessions)
                try:
                    # Cluster 1: Globe Search & Coordinate / Epoch / MOST Lookup
                    for q in user_rng.sample(SEARCH_QUERIES, k=persona["searches"]):
                        await _timed_get(
                            client,
                            "/api/v1/catalogs/search",
                            {"q": q, "limit": 12},
                            records,
                            "search",
                        )
                    if persona["check_object_epochs"]:
                        slug0, ra0, dec0 = user_rng.choice(hot_pool)
                        await _timed_get(
                            client,
                            "/api/v1/catalogs/object-epochs",
                            {
                                "object_id": f"showcase:{slug0}",
                                "label": slug0,
                                "target_slug": slug0,
                                "ra": ra0,
                                "dec": dec0,
                            },
                            records,
                            "object_epochs",
                        )
                    if persona["check_most"]:
                        await _timed_get(
                            client,
                            "/api/v1/solar-system/most",
                            {"obj_name": user_rng.choice(MOST_TARGETS)},
                            records,
                            "most",
                        )
                    await _pace_until(1)

                    # Cluster 2: Globe Panning & Zooming (FOV sweeps)
                    for _ in range(persona["fov_sweeps"]):
                        pra = round(user_rng.uniform(0.0, 359.5), 3)
                        pdec = round(user_rng.uniform(-75.0, 75.0), 3)
                        await _timed_get(
                            client,
                            "/api/v1/solar-system/in-fov",
                            {"ra": pra, "dec": pdec, "fov": user_rng.choice([5.0, 15.0, 45.0])},
                            records,
                            "in_fov",
                        )
                    await _pace_until(2)

                    # Cluster 3: Cold Cache Navigation (1-2 Random Non-Overlapping Sky Coords)
                    for c_offset in range(persona["cold_patches"]):
                        c_ra, c_dec = cold_coords[(user_idx * 2 + c_offset) % len(cold_coords)]
                        await _fetch_patch_and_frames(
                            client,
                            c_ra,
                            c_dec,
                            records,
                            endpoint_tag="cold",
                            fetch_spectrum=(user_rng.random() < persona["spectra_prob"]),
                        )
                    await _pace_until(3)

                    # Cluster 4: Purposeful Hot Navigation (first half of hot targets)
                    hot_picks = user_rng.sample(hot_pool, k=persona["hot_patches"])
                    half = max(1, len(hot_picks) // 2)
                    for h_idx, (_slug, h_ra, h_dec) in enumerate(hot_picks[:half]):
                        await _fetch_patch_and_frames(
                            client,
                            h_ra,
                            h_dec,
                            records,
                            endpoint_tag="hot",
                            fetch_spectrum=(h_idx == 0 and user_rng.random() < persona["spectra_prob"]),
                        )
                    await _pace_until(4)

                    # Cluster 5: Remaining Hot Navigation
                    for h_idx, (_slug, h_ra, h_dec) in enumerate(hot_picks[half:]):
                        await _fetch_patch_and_frames(
                            client,
                            h_ra,
                            h_dec,
                            records,
                            endpoint_tag="hot",
                            fetch_spectrum=(h_idx == 0 and user_rng.random() < persona["spectra_prob"]),
                        )
                    await _pace_until(5)

                    # Cluster 6: Predictable Randomness (3 random cities from 60-city pool)
                    picked_cities = user_rng.sample(cities_60, k=3)
                    zenith_targets: list[tuple[float, float]] = []
                    for city in picked_cities:
                        er = await _timed_get(
                            client,
                            "/api/v1/earth-to-sky",
                            {
                                "lat": city["lat"],
                                "lon": city["lon"],
                                "iso_time": "2026-11-21T12:00:00Z",
                            },
                            records,
                            "earth_to_sky",
                        )
                        if er is not None and er.status_code == 200:
                            z = er.json().get("zenith_icrs", {})
                            zenith_targets.append((float(z["ra"]), float(z["dec"])))

                    for z_ra, z_dec in zenith_targets[: persona["city_zenith_patches"]]:
                        await _fetch_patch_and_frames(
                            client,
                            z_ra,
                            z_dec,
                            records,
                            endpoint_tag="city_zenith",
                            fetch_spectrum=(user_rng.random() < persona["spectra_prob"]),
                        )
                    await _pace_until(6)
                finally:
                    stay_times_s.append(time.perf_counter() - user_t0)
                    active_sessions -= 1

        await asyncio.gather(*(_virtual_user(i, d) for i, d in enumerate(arrival_offsets)))
        elapsed_s = time.perf_counter() - phase_t0
        await asyncio.sleep(1.0)
        after_resp = await client.get("/internal/bench-cache-stats")
        after_resp.raise_for_status()
        after_cache = after_resp.json()

    with sampler._lock:
        proc_samples = list(sampler.samples)

    delta_tiers: dict[str, int] = {}
    all_keys = set(before_cache["tier_outcomes"]) | set(after_cache["tier_outcomes"])
    for k in sorted(all_keys):
        delta_tiers[k] = after_cache["tier_outcomes"].get(k, 0) - before_cache["tier_outcomes"].get(
            k, 0
        )

    delta_writes = {
        k: after_cache["cold_writes"].get(k, 0) - before_cache["cold_writes"].get(k, 0)
        for k in ("sky_patch", "spectrum", "zenith")
    }

    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ep[r["endpoint"]].append(r)

    endpoint_summary: dict[str, Any] = {}
    for ep, rows in sorted(by_ep.items()):
        lat_ms = [r["ms"] for r in rows]
        wire_b = sum(r["wire_bytes"] for r in rows)
        raw_b = sum(r["raw_bytes"] for r in rows)
        errors = sum(1 for r in rows if r["status"] >= 400)
        timeouts = sum(1 for r in rows if r["status"] == 599)
        over_45s = sum(1 for r in rows if r["ms"] >= 45_000.0)
        endpoint_summary[ep] = {
            "count": len(rows),
            "errors": errors,
            "timeouts": timeouts,
            "over_45s": over_45s,
            "mean_ms": round(statistics.mean(lat_ms), 2) if lat_ms else 0.0,
            "p50_ms": round(_percentile(lat_ms, 50), 2),
            "p95_ms": round(_percentile(lat_ms, 95), 2),
            "p99_ms": round(_percentile(lat_ms, 99), 2),
            "max_ms": round(max(lat_ms), 2),
            "wire_mb": round(wire_b / (1024 * 1024), 3),
            "raw_mb": round(raw_b / (1024 * 1024), 3),
        }

    overall_latency_ms = _compute_weighted_latency_summary(endpoint_summary)
    heavy_endpoints_latency_ms = _compute_weighted_latency_summary(
        endpoint_summary,
        prefixes=("sky_patch:", "spectrum:", "earth_to_sky", "object_epochs"),
    )

    cpu_vals = [s["cpu_pct_total"] for s in proc_samples]
    rss_vals = [s["rss_mb"] for s in proc_samples]
    hwm_vals = [s["hwm_mb"] for s in proc_samples]
    db_vals = [s["db_disk_mb"] for s in proc_samples]

    io_read_mb = (
        (proc_samples[-1]["io_read_bytes"] - proc_samples[0]["io_read_bytes"]) / (1024 * 1024)
        if len(proc_samples) >= 2
        else 0.0
    )
    io_write_mb = (
        (proc_samples[-1]["io_write_bytes"] - proc_samples[0]["io_write_bytes"]) / (1024 * 1024)
        if len(proc_samples) >= 2
        else 0.0
    )
    total_wire_mb = sum(r["wire_bytes"] for r in records) / (1024 * 1024)
    total_raw_mb = sum(r["raw_bytes"] for r in records) / (1024 * 1024)
    norm_cores = max(1, vcpus)

    return {
        "phase": phase_name,
        "seed": seed,
        "sessions": total_sessions,
        "arrival_window_s": duration_s,
        "elapsed_s": round(elapsed_s, 2),
        "total_requests": len(records),
        "requests_per_sec": round(len(records) / max(1e-3, elapsed_s), 2),
        "max_concurrent_sessions": max_concurrent,
        "avg_concurrent_sessions": round(statistics.mean(concurrency_samples), 2)
        if concurrency_samples
        else 0.0,
        "stay_time_s": {
            "min": round(min(stay_times_s), 2) if stay_times_s else 0.0,
            "p50": round(_percentile(stay_times_s, 50), 2) if stay_times_s else 0.0,
            "mean": round(statistics.mean(stay_times_s), 2) if stay_times_s else 0.0,
            "p95": round(_percentile(stay_times_s, 95), 2) if stay_times_s else 0.0,
            "max": round(max(stay_times_s), 2) if stay_times_s else 0.0,
        },
        "personas": dict(persona_counts),
        "overall_latency_ms": overall_latency_ms,
        "heavy_endpoints_latency_ms": heavy_endpoints_latency_ms,
        "server_cpu": {
            "pinned_vcpus": norm_cores,
            "pinned_cpu_list": select_physical_first_cpus(norm_cores),
            "coadd_workers": compute_dynamic_coadd_workers(norm_cores),
            "avg_pct_1vcpu_norm": round(statistics.mean(cpu_vals), 1) if cpu_vals else 0.0,
            "p95_pct_1vcpu_norm": round(_percentile(cpu_vals, 95), 1) if cpu_vals else 0.0,
            "peak_pct_1vcpu_norm": round(max(cpu_vals), 1) if cpu_vals else 0.0,
            "avg_pct_allocated_vcpus": round(statistics.mean(cpu_vals) / norm_cores, 1)
            if cpu_vals
            else 0.0,
            "peak_pct_allocated_vcpus": round(max(cpu_vals) / norm_cores, 1) if cpu_vals else 0.0,
            "avg_pct_4vcpu_norm": round(statistics.mean(cpu_vals) / 4.0, 1) if cpu_vals else 0.0,
            "peak_pct_4vcpu_norm": round(max(cpu_vals) / 4.0, 1) if cpu_vals else 0.0,
        },
        "server_ram": {
            "start_rss_mb": round(rss_vals[0], 1) if rss_vals else 0.0,
            "avg_rss_mb": round(statistics.mean(rss_vals), 1) if rss_vals else 0.0,
            "peak_rss_mb": round(max(rss_vals), 1) if rss_vals else 0.0,
            "peak_hwm_mb": round(max(hwm_vals), 1) if hwm_vals else 0.0,
        },
        "disk_and_sqlite": {
            "io_read_mb": round(io_read_mb, 2),
            "io_write_mb": round(io_write_mb, 2),
            "start_db_mb": round(db_vals[0], 2) if db_vals else 0.0,
            "end_db_mb": round(db_vals[-1], 2) if db_vals else 0.0,
            "sqlite_max_mb": round(after_cache["sqlite_max_bytes"] / (1024 * 1024), 1),
            "sqlite_tables_after": after_cache["sqlite_tables"],
        },
        "network": {
            "total_wire_mb": round(total_wire_mb, 2),
            "total_uncompressed_mb": round(total_raw_mb, 2),
            "compression_ratio": round(total_raw_mb / max(1e-6, total_wire_mb), 2),
            "mb_per_session_wire": round(total_wire_mb / max(1, total_sessions), 2),
            "avg_egress_mbps": round((total_wire_mb * 8.0) / max(1e-3, elapsed_s), 2),
            "projected_gb_per_hour_at_20_conn_min": round(
                (total_wire_mb / max(1, total_sessions)) * 1200.0 / 1024.0, 3
            ),
        },
        "application_caches": {
            "mem_lru_raw_hits_delta": after_cache["mem_lru_raw_hits"]
            - before_cache["mem_lru_raw_hits"],
            "mem_lru_raw_misses_delta": after_cache["mem_lru_raw_misses"]
            - before_cache["mem_lru_raw_misses"],
            "mem_lru_used_mb_after": round(after_cache["mem_lru_current_bytes"] / (1024 * 1024), 2),
            "mem_lru_max_mb": round(after_cache["mem_lru_max_bytes"] / (1024 * 1024), 1),
            "mem_lru_entries_after": after_cache["mem_lru_entries"],
            "tier_outcomes_delta": delta_tiers,
            "cold_writes_delta": delta_writes,
            "cold_coadd_calls_delta": after_cache["cold_coadd_calls"]
            - before_cache["cold_coadd_calls"],
            "cold_spectrum_calls_delta": after_cache["cold_spectrum_calls"]
            - before_cache["cold_spectrum_calls"],
        },
        "endpoints": endpoint_summary,
    }


def run_single_configuration(
    cfg: dict[str, Any],
    hot_pool: list[tuple[str, float, float]],
    cities_60: list[dict[str, Any]],
    cold_run1: list[tuple[float, float]],
    cold_run2: list[tuple[float, float]],
    *,
    db_path: Path,
    total_sessions: int,
    duration_s: float,
    max_concurrent_users: int,
    smoke_test: bool,
) -> dict[str, Any]:
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink()

    port = _pick_free_port(18080)
    base_url = f"http://127.0.0.1:{port}"
    cfg_name = cfg["name"]
    vcpus = int(cfg["vcpus"])
    coadd_workers = int(cfg.get("coadd_workers") or compute_dynamic_coadd_workers(vcpus))
    in_memory_mb = int(cfg["in_memory_cache_mb"])
    sqlite_max_mb = int(cfg["sqlite_max_cache_mb"])

    server_log_path = BENCH_DIR / f"bench_server_{cfg_name}.log"
    server_log_fh = server_log_path.open("w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            "--port",
            str(port),
            "--db",
            str(db_path),
            "--vcpus",
            str(vcpus),
            "--coadd-workers",
            str(coadd_workers),
            "--in-memory-mb",
            str(in_memory_mb),
            "--sqlite-max-mb",
            str(sqlite_max_mb),
        ],
        stdout=server_log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        import urllib.request

        ready = False
        for _ in range(100):
            if server_proc.poll() is not None:
                server_log_fh.flush()
                log_tail = server_log_path.read_text(encoding="utf-8")[-2000:]
                raise RuntimeError(f"Benchmark server exited early:\n{log_tail}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1) as resp:
                    if resp.status == 200:
                        with urllib.request.urlopen(
                            f"{base_url}/internal/bench-cache-stats", timeout=2
                        ) as sresp:
                            if sresp.status == 200:
                                ready = True
                                break
            except Exception:
                time.sleep(0.15)
        if not ready:
            raise RuntimeError("Timed out waiting for benchmark server /health")

        sampler = ProcTelemetrySampler(server_proc.pid, db_path, base_url)
        sampler.start()

        async def _execute_pair() -> tuple[dict[str, Any], dict[str, Any]]:
            print(
                f"[{cfg_name}] Run 1 (Cold Start, seed=42, {total_sessions} sessions over {duration_s}s, "
                f"vcpus={vcpus} cpus={select_physical_first_cpus(vcpus)}, coadd_workers={coadd_workers}, "
                f"ram_lru={in_memory_mb}MB, sqlite={sqlite_max_mb}MB)...",
                flush=True,
            )
            r1 = await run_phase(
                "run_1_cold_start",
                seed=42,
                base_url=base_url,
                hot_pool=hot_pool,
                cities_60=cities_60,
                cold_coords=cold_run1,
                sampler=sampler,
                total_sessions=total_sessions,
                duration_s=duration_s,
                max_concurrent_users=max_concurrent_users,
                vcpus=vcpus,
                smoke_test=smoke_test,
            )
            print(
                f"[{cfg_name}] Run 1 done in {r1['elapsed_s']}s | mean_ms={r1['overall_latency_ms']['mean_ms']}ms | "
                f"w_p50={r1['overall_latency_ms']['weighted_p50_ms']}ms | peak_rss={r1['server_ram']['peak_rss_mb']}MB | "
                f"lru={r1['application_caches']['mem_lru_used_mb_after']}MB | db={r1['disk_and_sqlite']['end_db_mb']}MB",
                flush=True,
            )

            await asyncio.sleep(2.0)
            print(
                f"[{cfg_name}] Run 2 (Prewarmed, seed=1337, {total_sessions} sessions over {duration_s}s)...",
                flush=True,
            )
            r2 = await run_phase(
                "run_2_prewarmed",
                seed=1337,
                base_url=base_url,
                hot_pool=hot_pool,
                cities_60=cities_60,
                cold_coords=cold_run2,
                sampler=sampler,
                total_sessions=total_sessions,
                duration_s=duration_s,
                max_concurrent_users=max_concurrent_users,
                vcpus=vcpus,
                smoke_test=smoke_test,
            )
            print(
                f"[{cfg_name}] Run 2 done in {r2['elapsed_s']}s | mean_ms={r2['overall_latency_ms']['mean_ms']}ms | "
                f"w_p50={r2['overall_latency_ms']['weighted_p50_ms']}ms | peak_rss={r2['server_ram']['peak_rss_mb']}MB | "
                f"lru={r2['application_caches']['mem_lru_used_mb_after']}MB | db={r2['disk_and_sqlite']['end_db_mb']}MB",
                flush=True,
            )
            return r1, r2

        run1, run2 = asyncio.run(_execute_pair())
        sampler.stop()
        return {
            "config": cfg,
            "backend_run_1": run1,
            "backend_run_2": run2,
        }
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        server_log_fh.close()


def _parse_configs_arg(raw_configs: str) -> list[dict[str, Any]]:
    """Parse 'name:vcpus:coadd_workers:in_mem_mb:sqlite_mb,...' into config dicts."""
    parsed: list[dict[str, Any]] = []
    for item in raw_configs.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid --configs entry '{item}'; expected name:vcpus:coadd_workers:in_mem_mb:sqlite_mb"
            )
        name, vc, cw, mem_mb, db_mb = parts
        vcpus_int = int(vc)
        cw_int = int(cw) if int(cw) > 0 else compute_dynamic_coadd_workers(vcpus_int)
        parsed.append(
            {
                "name": name,
                "fly_equivalent": f"{vcpus_int} vCPU ({cw_int} coadd) / {mem_mb}MB LRU / {db_mb}MB SQLite",
                "vcpus": vcpus_int,
                "coadd_workers": cw_int,
                "pinned_cpus": select_physical_first_cpus(vcpus_int),
                "in_memory_cache_mb": int(mem_mb),
                "sqlite_max_cache_mb": int(db_mb),
            }
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--probe-resources", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="Run the auto-probed multi-config matrix")
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated name:vcpus:coadd_workers:in_mem_mb:sqlite_mb configs (0 coadd_workers = dynamic formula)",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--vcpus", type=int, default=4)
    parser.add_argument("--coadd-workers", type=int, default=0, help="0 = dynamic formula")
    parser.add_argument("--in-memory-mb", type=int, default=0, help="0 = auto-probe safe max")
    parser.add_argument("--sqlite-max-mb", type=int, default=0, help="0 = auto-probe safe max")
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--max-concurrent", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.serve:
        run_bench_server(
            port=args.port,
            db_path=args.db,
            vcpus=args.vcpus,
            coadd_workers=args.coadd_workers,
            in_memory_mb=args.in_memory_mb or 512,
            sqlite_max_mb=args.sqlite_max_mb or 750,
        )
        return 0

    wsl_probe = probe_wsl_resources(args.db)
    if args.probe_resources:
        print(json.dumps(wsl_probe, indent=2), flush=True)
        return 0

    sys.path.insert(0, str(BACKEND_DIR))
    from app.models.contracts import SHOWCASE_TARGETS

    hot_pool = [(slug, float(m["ra"]), float(m["dec"])) for slug, m in SHOWCASE_TARGETS.items()]
    hot_pool.extend(HOT_EXTRA_COORDS)
    cities_60 = _load_60_cities()

    total_sessions = 4 if args.smoke_test else args.sessions
    duration_s = 3.0 if args.smoke_test else args.duration_s

    forbidden = [(ra, dec) for _, ra, dec in hot_pool]
    cold_run1 = _generate_cold_coords(seed=42, count=total_sessions * 2, forbidden=forbidden)
    cold_run2 = _generate_cold_coords(
        seed=1337, count=total_sessions * 2, forbidden=forbidden + cold_run1
    )

    if args.configs:
        configs = _parse_configs_arg(args.configs)
    elif args.matrix:
        configs = list(wsl_probe["recommended_matrix"])
    else:
        in_mem = args.in_memory_mb or wsl_probe["max_safe_in_memory_mb"]
        sq_max = args.sqlite_max_mb or wsl_probe["max_safe_sqlite_mb"]
        cw = args.coadd_workers or compute_dynamic_coadd_workers(args.vcpus)
        configs = [
            {
                "name": f"cfg_{args.vcpus}vcpu_{in_mem}m_ram",
                "fly_equivalent": f"shared-cpu-{args.vcpus}x ({cw} coadd, {in_mem}MB LRU / {sq_max}MB SQLite)",
                "vcpus": args.vcpus,
                "coadd_workers": cw,
                "pinned_cpus": select_physical_first_cpus(args.vcpus),
                "in_memory_cache_mb": in_mem,
                "sqlite_max_cache_mb": sq_max,
            }
        ]

    config_runs: list[dict[str, Any]] = []
    for cfg in configs:
        res = run_single_configuration(
            cfg,
            hot_pool=hot_pool,
            cities_60=cities_60,
            cold_run1=cold_run1,
            cold_run2=cold_run2,
            db_path=args.db,
            total_sessions=total_sessions,
            duration_s=duration_s,
            max_concurrent_users=args.max_concurrent,
            smoke_test=args.smoke_test,
        )
        config_runs.append(res)

    primary = config_runs[-1]
    output_payload = {
        "wsl_resources_at_start": wsl_probe,
        "configuration_runs": config_runs,
        "backend_run_1": primary["backend_run_1"],
        "backend_run_2": primary["backend_run_2"],
    }
    if not args.smoke_test:
        RESULTS_PATH.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(f"Saved multi-config benchmark results to {RESULTS_PATH}", flush=True)
    else:
        print("Smoke test completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
