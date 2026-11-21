#!/usr/bin/env python3
"""6×4 Step-Load & Cache-Saturation Stress Test (C = 2..40, 50%/85%/100%/125% Saturation).

Default sizing targets our 4 vCPU / 4 GB RAM / 6 GB Volume Fly.io ceiling:
- IN_MEMORY_CACHE_MB = 1536 (with optional --auto-clamp-wsl on smaller dev hosts)
- SQLITE_MAX_CACHE_MB = 4096
- 6 concurrency stages C in [2, 6, 12, 20, 30, 40]
- 4 cache saturation tiers per stage: [50%, 85%, 100%, 125%] of SQLite capacity
- 120 unique spawned users per saturation tier (480 users/stage, 2,880 users total)
- Cache carries across 50% -> 85% -> 100% -> 125% within a concurrency stage, and resets cleanly
  between concurrency stages so every concurrency level gets an identical saturation sweep.
"""

from __future__ import annotations

import argparse
import asyncio
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

import numpy as np

BENCH_DIR = Path("/home/line/hackathons/star-hunters-workspace/benchmarks")
DATASET_DIR = BENCH_DIR / "stress_dataset"
BACKEND_DIR = Path("/home/line/hackathons/star-hunters-workspace/star-hunters-main/backend")
DB_PATH = BENCH_DIR / "stress_cache.db"
RESULTS_PATH = BENCH_DIR / "stress_test_results.json"
REPORT_PATH = BENCH_DIR / "stress_test_report.md"

DEFAULT_CONCURRENCIES = [2, 6, 12, 20, 30, 40]
DEFAULT_SATURATION_TIERS = [50, 85, 100, 125]


def compute_dynamic_coadd_workers(cpu_cores: int) -> int:
    """If <= 6 cores reserve half for IO/asyncio; if > 6 cores reserve max(3, ceil(total/4))."""
    total = max(1, int(cpu_cores))
    reserved = max(1, math.ceil(total / 2)) if total <= 6 else max(3, math.ceil(total / 4))
    return max(1, total - reserved)


def select_physical_first_cpus(vcpus: int) -> list[int]:
    """Select logical CPUs prioritizing distinct physical core_ids before SMT siblings."""
    total_cores = os.cpu_count() or 4
    target = min(max(1, vcpus), total_cores)
    by_core_id: dict[int, list[int]] = defaultdict(list)
    try:
        for cpu_idx in range(total_cores):
            cid_path = Path(f"/sys/devices/system/cpu/cpu{cpu_idx}/topology/core_id")
            cid = int(cid_path.read_text(encoding="utf-8").strip()) if cid_path.exists() else cpu_idx
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


def _probe_avail_ram_and_disk(db_path: Path) -> tuple[int, float]:
    avail_ram_mb = 2048
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                avail_ram_mb = int(line.split()[1]) // 1024
                break
    st = os.statvfs(BENCH_DIR)
    free_disk_mb = (st.f_bavail * st.f_frsize) / (1024 * 1024)
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            free_disk_mb += p.stat().st_size / (1024 * 1024)
    return avail_ram_mb, free_disk_mb


def _pick_free_port(preferred: int = 18090) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (1.0 - (idx - lo)) + ordered[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Dedicated Instrumented Stress Server (--serve)
# ---------------------------------------------------------------------------


def run_stress_server(
    port: int,
    db_path: Path,
    dataset_dir: Path,
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

    resolved_coadds = coadd_workers if coadd_workers > 0 else compute_dynamic_coadd_workers(vcpus)

    sys.path.insert(0, str(BACKEND_DIR))
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    os.environ["ENABLE_ARCHIVAL_DATA"] = "true"
    os.environ["IN_MEMORY_CACHE_MB"] = str(in_memory_mb)
    os.environ["ARCHIVAL_MAX_CONCURRENT_COADDS"] = str(resolved_coadds)
    os.environ["COLD_MISS_RATE_LIMIT_PER_IP_PER_MIN"] = "0"
    os.environ["LOG_LEVEL"] = "WARNING"

    import uvicorn
    from app.main import create_app
    from app.services import backend_service as bs_mod
    from app.services import catalog_clients as cc_mod
    from app.services import celestial_catalog_pipeline as ccp_mod
    from app.services import object_epochs as oe_mod
    from app.services.backend_service import StarHuntersBackendService
    from app.services.spatial_cache import SpatialSqliteCache, compute_grid_key
    from app.services.spectrum_extractor import extract_forced_spectrum
    from app.services.spherex_fits import SpherexCutout, SpherexObservation
    from app.services.spherex_coadd import build_target_wcs_header
    from app.utils.astronomy import datetime_to_mjd

    patches_data = json.loads((dataset_dir / "sky_patches_catalog.json").read_text(encoding="utf-8"))
    by_grid_key = {p["grid_key"]: p for p in patches_data}
    templates = np.load(dataset_dir / "cutout_templates.npz")
    tpl_images = templates["clean_images"]
    tpl_vars = templates["variances"]
    tpl_flags = templates["flags_masks"]

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
                    "source": "jpl_horizons_stress",
                }
            )
    cache.upsert_ephemeris_points(ephem_points)

    stats_lock = threading.Lock()
    stats: dict[str, Any] = {
        "mem_lru_raw_hits": 0,
        "mem_lru_raw_misses": 0,
        "mem_lru_evictions": 0,
        "sqlite_evictions": 0,
        "tier_outcomes": defaultdict(int),
        "cold_writes": defaultdict(int),
        "cold_coadd_calls": 0,
        "cold_spectrum_calls": 0,
    }

    orig_mem_get = cache._mem_cache.get
    orig_mem_put = cache._mem_cache.put

    def tracked_mem_get(key: str) -> bytes | None:
        res = orig_mem_get(key)
        with stats_lock:
            if res is not None:
                stats["mem_lru_raw_hits"] += 1
            else:
                stats["mem_lru_raw_misses"] += 1
        return res

    def tracked_mem_put(key: str, value: bytes) -> None:
        before_entries = cache._mem_cache.entry_count
        existed = key in cache._mem_cache._entries
        orig_mem_put(key, value)
        after_entries = cache._mem_cache.entry_count
        evicted = max(0, (before_entries + (0 if existed else 1)) - after_entries)
        if evicted > 0:
            with stats_lock:
                stats["mem_lru_evictions"] += evicted

    cache._mem_cache.get = tracked_mem_get  # type: ignore[method-assign]
    cache._mem_cache.put = tracked_mem_put  # type: ignore[method-assign]

    orig_prune = cache._prune_if_needed

    def tracked_prune(conn: Any, preserve_grid_key: str | None = None) -> None:
        before_cnt = int(conn.execute("SELECT COUNT(*) FROM sky_patches").fetchone()[0])
        orig_prune(conn, preserve_grid_key=preserve_grid_key)
        after_cnt = int(conn.execute("SELECT COUNT(*) FROM sky_patches").fetchone()[0])
        if before_cnt > after_cnt:
            with stats_lock:
                stats["sqlite_evictions"] += before_cnt - after_cnt

    cache._prune_if_needed = tracked_prune  # type: ignore[method-assign]

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

    def dataset_cutout_fetcher(
        ra: float,
        dec: float,
        fov_arcmin: float = 5.0,
        max_cutouts: int | None = None,
        **_kwargs: Any,
    ) -> list[SpherexCutout]:
        gk, _, _ = compute_grid_key(ra, dec, fov_arcmin)
        pmeta = by_grid_key.get(gk)
        tid = int(pmeta["template_id"]) if pmeta else int(abs(round(ra * 10 + dec * 10))) % 32
        fscale = float(pmeta["flux_scale"]) if pmeta else 1.0
        mjds = (60500.0,) if (max_cutouts is not None and max_cutouts <= 6) else (60500.0, 60682.5)

        header = build_target_wcs_header(ra, dec, fov_arcmin / 60.0)
        # Adjust FITS header to match 64x64 cutout scale
        header = dict(header)
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        header["CRPIX1"] = 32.5
        header["CRPIX2"] = 32.5
        header["CDELT1"] = -(fov_arcmin / 60.0) / 64.0
        header["CDELT2"] = (fov_arcmin / 60.0) / 64.0

        cutouts: list[SpherexCutout] = []
        for ep_idx, mjd in enumerate(mjds):
            for det in range(1, 7):
                img = (tpl_images[tid, ep_idx, det - 1] * fscale).astype(np.float32, copy=True)
                var = tpl_vars[tid, ep_idx, det - 1].astype(np.float32, copy=True)
                flg = tpl_flags[tid, ep_idx, det - 1].astype(np.uint32, copy=True)
                wmap = 1.0 / np.maximum(var, 1e-4)
                cutouts.append(
                    SpherexCutout(
                        clean_image=img,
                        variance=var,
                        weight_map=wmap,
                        flags_mask=flg,
                        mjd_obs=mjd,
                        bandpass=f"SPHEREx-D{det}",
                        uri=f"stress://{gk}/ep{ep_idx}_d{det}.fits",
                        source="stress_dataset",
                        header=header,
                    )
                )
        with stats_lock:
            if len(mjds) == 2:
                stats["cold_coadd_calls"] += 1
            else:
                stats["cold_spectrum_calls"] += 1
        return cutouts

    def disabled_fallback(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("External archive fallback disabled in offline stress test")

    async def dataset_gaia_fetch(
        ra_deg: float, dec_deg: float, *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict[str, Any]], str]:
        gk, _, _ = compute_grid_key(ra_deg, dec_deg, 5.0)
        pmeta = by_grid_key.get(gk)
        return (list(pmeta["gaia_rows"]) if pmeta else []), "stress_gaia"

    async def dataset_simbad_fetch(
        ra_deg: float, dec_deg: float, *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict[str, Any]], str]:
        gk, _, _ = compute_grid_key(ra_deg, dec_deg, 5.0)
        pmeta = by_grid_key.get(gk)
        return (list(pmeta["simbad_rows"]) if pmeta else []), "stress_simbad"

    async def no_live_horizons(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def dataset_most(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "query_name": kwargs.get("obj_name", ""),
            "status": "ok",
            "matches": [
                {"mjd_start": 60500.0, "ra": 150.0, "dec": 12.0},
                {"mjd_start": 60682.5, "ra": 150.4, "dec": 12.1},
            ],
        }

    def dataset_obs_query(
        ra: float, dec: float, *_args: Any, **_kwargs: Any
    ) -> list[SpherexObservation]:
        return [
            SpherexObservation(uri=f"stress://obs1_{ra:.2f}", bandpass="SPHEREx-D1", mjd_obs=60500.0),
            SpherexObservation(uri=f"stress://obs2_{dec:.2f}", bandpass="SPHEREx-D2", mjd_obs=60682.5),
        ]

    ccp_mod.fetch_simbad_objects = dataset_simbad_fetch  # type: ignore[assignment]
    ccp_mod.fetch_gaia_dr3_objects = dataset_gaia_fetch  # type: ignore[assignment]
    bs_mod.fetch_most_spherex_matches = dataset_most  # type: ignore[assignment]
    oe_mod.query_observations = dataset_obs_query  # type: ignore[assignment]

    service = StarHuntersBackendService(
        cache=cache,
        enable_archival_data=True,
        wise_fallback_fetcher=disabled_fallback,
        skyview_fallback_fetcher=disabled_fallback,
        spherex_cutout_fetcher=dataset_cutout_fetcher,
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
            mem_evict = int(stats["mem_lru_evictions"])
            sq_evict = int(stats["sqlite_evictions"])
            coadd_calls = int(stats["cold_coadd_calls"])
            spec_calls = int(stats["cold_spectrum_calls"])

        with cache.connection() as conn:
            patch_row = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(byte_size), 0) AS bytes FROM sky_patches"
            ).fetchone()
            frame_row = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(LENGTH(compressed_blob)), 0) AS comp_bytes "
                "FROM binary_frames"
            ).fetchone()
            spec_row = conn.execute("SELECT COUNT(*) AS cnt FROM spectra_cache").fetchone()
            zen_row = conn.execute("SELECT COUNT(*) AS cnt FROM capital_zenith_cache").fetchone()

        return {
            "mem_lru_raw_hits": raw_hits,
            "mem_lru_raw_misses": raw_misses,
            "mem_lru_evictions": mem_evict,
            "sqlite_evictions": sq_evict,
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
                "spectra_count": int(spec_row["cnt"]),
                "zenith_count": int(zen_row["cnt"]),
            },
        }

    app.router.routes.insert(0, app.router.routes.pop())

    @app.post("/internal/bench-reset-cache")
    async def bench_reset_cache() -> dict[str, str]:
        service.cancel_background_tasks()
        with cache.connection() as conn:
            conn.execute("DELETE FROM binary_frames")
            conn.execute("DELETE FROM sky_patches")
            conn.execute("DELETE FROM spectra_cache")
            conn.execute("DELETE FROM capital_zenith_cache")
            conn.execute("DELETE FROM catalog_region_cache")
        cache._mem_cache.clear()
        return {"status": "reset"}

    app.router.routes.insert(0, app.router.routes.pop())

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------------------
# /proc/<pid> Telemetry Sampler
# ---------------------------------------------------------------------------


class ProcTelemetrySampler:
    def __init__(self, pid: int, db_path: Path) -> None:
        self.pid = pid
        self.db_path = db_path
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
        return (int(fields[11]) + int(fields[12])) / self.clk_tck

    def _read_proc_status(self) -> tuple[float, float, int]:
        rss_kb = hwm_kb = 0.0
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

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
                prev_cpu, prev_wall = now_cpu, now_wall
                rss_mb, hwm_mb, threads = self._read_proc_status()
                io_stats = self._read_proc_io()
                db_mb = self._db_disk_bytes() / (1024.0 * 1024.0)
                with self._lock:
                    self.samples.append(
                        {
                            "cpu_pct_total": round(cpu_pct, 2),
                            "rss_mb": round(rss_mb, 2),
                            "hwm_mb": round(hwm_mb, 2),
                            "threads": threads,
                            "io_read_bytes": io_stats.get("read_bytes", 0),
                            "io_write_bytes": io_stats.get("write_bytes", 0),
                            "db_disk_mb": round(db_mb, 3),
                        }
                    )
            except OSError:
                break


# ---------------------------------------------------------------------------
# Closed-Loop Step-Load Stage Execution
# ---------------------------------------------------------------------------


async def _timed_get(
    client: Any,
    url: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    endpoint: str,
    *,
    headers: dict[str, str] | None = None,
) -> Any | None:
    import httpx

    t0 = time.perf_counter()
    try:
        resp = await client.get(url, params=params, headers=headers)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "endpoint": endpoint,
                "status": resp.status_code,
                "ms": elapsed_ms,
                "wire_bytes": resp.num_bytes_downloaded,
                "raw_bytes": len(resp.content),
            }
        )
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


def _pick_patch_80_20(
    rng: random.Random,
    active_pool: list[dict[str, Any]],
    frontier_start_idx: int,
    frontier_deck: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drive tier saturation cleanly via frontier_deck while preserving hot-key reuse."""
    if frontier_deck and rng.random() < 0.85:
        return frontier_deck.pop()
    if rng.random() < 0.35 and frontier_start_idx < len(active_pool):
        # Frontier band newly unlocked in this tier
        return active_pool[rng.randint(frontier_start_idx, len(active_pool) - 1)]
    if rng.random() < 0.45:
        # Top 10% hot showcase/popular pool
        top_n = max(1, len(active_pool) // 10)
        return active_pool[rng.randint(0, top_n - 1)]
    return active_pool[rng.randint(0, len(active_pool) - 1)]


async def _execute_user_journey(
    client: Any,
    user_rng: random.Random,
    archetype: dict[str, Any],
    active_pool: list[dict[str, Any]],
    frontier_start_idx: int,
    frontier_deck: list[dict[str, Any]],
    action_buckets: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
) -> int:
    """Execute 100-250 curated actions (zero timer sleeps) matching the chosen archetype."""
    target_actions = user_rng.randint(archetype["min_actions"], archetype["max_actions"])
    name = archetype["name"]
    actions_done = 0

    search_acts = action_buckets["/api/v1/catalogs/search"]
    e2s_acts = action_buckets["/api/v1/earth-to-sky"]
    fov_acts = action_buckets["/api/v1/solar-system/in-fov"]
    epoch_acts = action_buckets["/api/v1/catalogs/object-epochs"]
    most_acts = action_buckets["/api/v1/solar-system/most"]
    frame_modes = action_buckets["/api/v1/frames"]

    # Thundering herd picks a shared patch across concurrent users in the wave
    herd_patch = active_pool[min(len(active_pool) - 1, frontier_start_idx)]

    while actions_done < target_actions:
        if name == "rapid_globe_frustum_thrasher":
            for _ in range(user_rng.randint(8, 14)):
                act = user_rng.choice(fov_acts)
                await _timed_get(client, act["endpoint"], act["params"], records, "in_fov")
                actions_done += 1
                if actions_done >= target_actions:
                    break
            act_s = user_rng.choice(search_acts)
            await _timed_get(client, act_s["endpoint"], act_s["params"], records, "search")
            actions_done += 1
            continue

        # 1. Pre-navigation discovery (Earth-to-Sky / Search / FOV / MOST)
        if name in ("zenith_to_blinker", "capital_city_hopper"):
            e2s = user_rng.choice(e2s_acts)
            await _timed_get(client, e2s["endpoint"], e2s["params"], records, "earth_to_sky")
            actions_done += 1
        elif name in ("search_blinker_spectra", "spectroscopy_lab_intensive"):
            sa = user_rng.choice(search_acts)
            await _timed_get(client, sa["endpoint"], sa["params"], records, "search")
            actions_done += 1
        elif name == "solar_system_most_tracker":
            ma = user_rng.choice(most_acts)
            await _timed_get(client, ma["endpoint"], ma["params"], records, "most")
            actions_done += 1
            ea = user_rng.choice(epoch_acts)
            await _timed_get(client, ea["endpoint"], ea["params"], records, "object_epochs")
            actions_done += 2

        # 2. Pick sky patch
        if name == "thundering_herd_coalescer":
            patch = herd_patch
        elif name == "deep_cold_sky_sweep":
            if frontier_deck:
                patch = frontier_deck.pop()
            else:
                patch = active_pool[user_rng.randint(frontier_start_idx, len(active_pool) - 1)]
        elif name == "polar_boundary_edge":
            patch = active_pool[-1 - (user_rng.randint(0, min(25, len(active_pool) - 1)))]
        else:
            patch = _pick_patch_80_20(user_rng, active_pool, frontier_start_idx, frontier_deck)

        ra, dec = patch["ra"], patch["dec"]

        # Stage A + Stage B
        await _timed_get(
            client,
            "/api/v1/catalogs",
            {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
            records,
            "catalogs",
        )
        pr = await _timed_get(
            client,
            "/api/v1/sky-patch",
            {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
            records,
            "sky_patch",
        )
        actions_done += 2

        if pr is not None and pr.status_code == 200:
            manifest = pr.json()
            fm = user_rng.choice(frame_modes)["params"]
            precision = "float32" if name == "legacy_float32_identity" else fm["precision"]
            headers = {"Accept-Encoding": "identity"} if name == "legacy_float32_identity" else None

            urls: list[tuple[str, dict[str, Any]]] = []
            for ep in manifest.get("epochs", []):
                if ep.get("detectors_data_url"):
                    urls.append((ep["detectors_data_url"], {"precision": precision}))
                if ep.get("flags_mask_url"):
                    urls.append((ep["flags_mask_url"], {}))
            if urls:
                await asyncio.gather(
                    *(
                        _timed_get(client, u, p, records, f"frame:{precision}", headers=headers)
                        for u, p in urls
                    )
                )
                actions_done += len(urls)

        # Spectrum call (heavier in spectroscopy_lab_intensive & search_blinker_spectra)
        spec_prob = 0.85 if name == "spectroscopy_lab_intensive" else 0.45
        if user_rng.random() < spec_prob and actions_done < target_actions:
            await _timed_get(
                client,
                "/api/v1/spectrum",
                {"ra": ra, "dec": dec, "label": patch["label"]},
                records,
                "spectrum",
            )
            actions_done += 1

        # Supplemental FOV pan steps inside session
        for _ in range(user_rng.randint(1, 3)):
            if actions_done >= target_actions:
                break
            fa = user_rng.choice(fov_acts)
            await _timed_get(client, fa["endpoint"], fa["params"], records, "in_fov")
            actions_done += 1

    return actions_done


async def run_stage(
    stage_spec: dict[str, Any],
    base_url: str,
    patches: list[dict[str, Any]],
    prev_pool_size: int,
    archetypes: list[dict[str, Any]],
    action_buckets: dict[str, list[dict[str, Any]]],
    sampler: ProcTelemetrySampler,
    *,
    users_per_stage: int,
    seed: int,
) -> tuple[dict[str, Any], int]:
    import httpx

    concurrency = int(stage_spec["concurrency"])
    pct = int(stage_spec["dataset_pct"])
    # Dataset expands across the active patch corpus (50% -> 85% -> 100% -> 125%)
    pool_size = max(10, min(len(patches), int(round(len(patches) * (pct / 125.0)))))
    active_pool = patches[:pool_size]
    frontier_start_idx = min(prev_pool_size, max(0, pool_size - 1))
    frontier_deck = list(patches[prev_pool_size:pool_size])
    deck_rng = random.Random(seed * 10_000 + int(stage_spec["stage"]) * 31 + pct)
    deck_rng.shuffle(frontier_deck)

    weights = [a["weight"] for a in archetypes]
    records: list[dict[str, Any]] = []
    flow_latencies_ms: dict[str, list[float]] = defaultdict(list)
    flow_counts: dict[str, int] = defaultdict(int)
    actions_executed = 0

    user_queue: asyncio.Queue[int] = asyncio.Queue()
    for uid in range(users_per_stage):
        user_queue.put_nowait(uid)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept-Encoding": "gzip"},
        limits=httpx.Limits(max_connections=max(32, concurrency * 4), max_keepalive_connections=concurrency * 2),
    ) as client:
        before_cache = (await client.get("/internal/bench-cache-stats")).json()
        sampler.reset_samples()
        t0 = time.perf_counter()

        async def _worker(worker_id: int) -> None:
            nonlocal actions_executed
            while True:
                try:
                    uid = user_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                urng = random.Random(
                    seed * 100_000 + int(stage_spec["stage"]) * 1000 + pct * 7 + uid
                )
                arch = urng.choices(archetypes, weights=weights, k=1)[0]
                ut0 = time.perf_counter()
                done = await _execute_user_journey(
                    client,
                    urng,
                    arch,
                    active_pool,
                    frontier_start_idx,
                    frontier_deck,
                    action_buckets,
                    records,
                )
                u_ms = (time.perf_counter() - ut0) * 1000.0
                flow_latencies_ms[arch["name"]].append(u_ms)
                flow_counts[arch["name"]] += 1
                actions_executed += done

        await asyncio.gather(*(_worker(w) for w in range(concurrency)))
        elapsed_s = time.perf_counter() - t0
        after_cache = (await client.get("/internal/bench-cache-stats")).json()

    with sampler._lock:
        proc_samples = list(sampler.samples)

    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ep[r["endpoint"]].append(r)

    endpoint_summary: dict[str, Any] = {}
    for ep, rows in sorted(by_ep.items()):
        lats = [r["ms"] for r in rows]
        errors = sum(1 for r in rows if r["status"] >= 400)
        endpoint_summary[ep] = {
            "count": len(rows),
            "errors": errors,
            "mean_ms": round(statistics.mean(lats), 2),
            "p50_ms": round(_percentile(lats, 50), 2),
            "p95_ms": round(_percentile(lats, 95), 2),
            "p99_ms": round(_percentile(lats, 99), 2),
            "max_ms": round(max(lats), 2),
        }

    all_lats = [r["ms"] for r in records]
    cpu_vals = [s["cpu_pct_total"] for s in proc_samples]
    rss_vals = [s["rss_mb"] for s in proc_samples]
    io_r_mb = (
        (proc_samples[-1]["io_read_bytes"] - proc_samples[0]["io_read_bytes"]) / (1024 * 1024)
        if len(proc_samples) >= 2
        else 0.0
    )
    io_w_mb = (
        (proc_samples[-1]["io_write_bytes"] - proc_samples[0]["io_write_bytes"]) / (1024 * 1024)
        if len(proc_samples) >= 2
        else 0.0
    )
    total_wire_mb = sum(r["wire_bytes"] for r in records) / (1024 * 1024)

    delta_tiers = {
        k: after_cache["tier_outcomes"].get(k, 0) - before_cache["tier_outcomes"].get(k, 0)
        for k in sorted(set(before_cache["tier_outcomes"]) | set(after_cache["tier_outcomes"]))
    }

    flow_summary = {
        fname: {
            "users": flow_counts[fname],
            "mean_journey_ms": round(statistics.mean(lats), 1),
            "p95_journey_ms": round(_percentile(lats, 95), 1),
        }
        for fname, lats in sorted(flow_latencies_ms.items())
    }

    res_count = max(1, int(after_cache["sqlite_tables"]["sky_patches_count"]))
    avg_patch_mb = round(
        (after_cache["sqlite_tables"]["sky_patches_bytes"] / (1024 * 1024)) / res_count,
        3,
    )

    return (
        {
            "stage": stage_spec["stage"],
            "tier": stage_spec.get("tier", 1),
            "concurrency": concurrency,
            "saturation_pct": pct,
            "dataset_pct_of_sqlite_cap": pct,
            "active_patch_pool": pool_size,
            "spawned_users": users_per_stage,
            "actions_executed": actions_executed,
            "total_http_requests": len(records),
            "elapsed_s": round(elapsed_s, 2),
            "throughput": {
                "requests_per_sec": round(len(records) / max(1e-3, elapsed_s), 2),
                "actions_per_sec": round(actions_executed / max(1e-3, elapsed_s), 2),
                "users_per_sec": round(users_per_stage / max(1e-3, elapsed_s), 2),
                "egress_mbps": round((total_wire_mb * 8.0) / max(1e-3, elapsed_s), 2),
            },
            "overall_http_latency_ms": {
                "mean_ms": round(statistics.mean(all_lats), 2) if all_lats else 0.0,
                "p50_ms": round(_percentile(all_lats, 50), 2),
                "p95_ms": round(_percentile(all_lats, 95), 2),
                "p99_ms": round(_percentile(all_lats, 99), 2),
                "max_ms": round(max(all_lats), 2) if all_lats else 0.0,
                "total_errors": sum(1 for r in records if r["status"] >= 400),
            },
            "resources": {
                "avg_cpu_1vcpu_pct": round(statistics.mean(cpu_vals), 1) if cpu_vals else 0.0,
                "peak_cpu_1vcpu_pct": round(max(cpu_vals), 1) if cpu_vals else 0.0,
                "peak_rss_mb": round(max(rss_vals), 1) if rss_vals else 0.0,
                "io_read_mb": round(io_r_mb, 2),
                "io_write_mb": round(io_w_mb, 2),
                "end_db_disk_mb": round(proc_samples[-1]["db_disk_mb"], 2) if proc_samples else 0.0,
            },
            "caches": {
                "mem_lru_used_mb": round(after_cache["mem_lru_current_bytes"] / (1024 * 1024), 2),
                "mem_lru_max_mb": round(after_cache["mem_lru_max_bytes"] / (1024 * 1024), 1),
                "mem_lru_evictions_delta": after_cache["mem_lru_evictions"]
                - before_cache["mem_lru_evictions"],
                "sqlite_patches_resident": after_cache["sqlite_tables"]["sky_patches_count"],
                "sqlite_logical_mb": round(
                    after_cache["sqlite_tables"]["sky_patches_bytes"] / (1024 * 1024), 2
                ),
                "avg_sqlite_mb_per_patch": avg_patch_mb,
                "sqlite_evictions_delta": after_cache["sqlite_evictions"]
                - before_cache["sqlite_evictions"],
                "cold_coadds_delta": after_cache["cold_coadd_calls"]
                - before_cache["cold_coadd_calls"],
                "tier_outcomes_delta": delta_tiers,
            },
            "endpoints": endpoint_summary,
            "user_flows": flow_summary,
        },
        pool_size,
    )


def build_stress_report(payload: dict[str, Any]) -> str:
    cfg = payload["config"]
    stages = payload["stages"]
    concurrencies = cfg.get("concurrencies", [])
    tiers = cfg.get("tiers", [])
    lines = [
        f"# {len(concurrencies)}×{len(tiers)} Step-Load & Cache-Saturation Stress Test Report",
        "",
        f"- **Target Configuration:** `{cfg['vcpus']}` vCPUs (`{cfg['coadd_workers']}` dynamic coadd workers), "
        f"`{cfg['in_memory_cache_mb']} MB` RAM LRU, `{cfg['sqlite_max_cache_mb']} MB` SQLite Cap",
        f"- **Dataset & Load:** `{cfg['dataset_patches']}` unique patches, "
        f"`{cfg['users_per_tier']}` spawned users per saturation tier (`{cfg['users_per_tier'] * len(stages)}` total users, 0 timer sleeps)",
        f"- **Concurrencies:** `{concurrencies}` · **Saturation Tiers:** `{tiers}` (`reset_between_stages={cfg.get('reset_between_stages', True)}`)",
        "",
        "## 1. Complete Concurrency × Saturation Matrix",
        "",
        "| Concurrency | Saturation | Pool | Reqs | Throughput (`req/s` · `act/s`) | Mean / p50 / p95 / p99 Latency (ms) | Errors | Avg/Peak CPU | Peak RSS | RAM Used (`Evicts`) | SQLite Patches (`Logical MB` · `Evicts`) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in stages:
        ol = s["overall_http_latency_ms"]
        tp = s["throughput"]
        rs = s["resources"]
        cc = s["caches"]
        lines.append(
            f"| **C={s['concurrency']}** | **{s['saturation_pct']}%** | {s['active_patch_pool']} | "
            f"{s['total_http_requests']} | {tp['requests_per_sec']} · {tp['actions_per_sec']} | "
            f"**{ol['mean_ms']}** / {ol['p50_ms']} / {ol['p95_ms']} / {ol['p99_ms']} | {ol['total_errors']} | "
            f"{rs['avg_cpu_1vcpu_pct']}% / {rs['peak_cpu_1vcpu_pct']}% | {rs['peak_rss_mb']} MB | "
            f"{cc['mem_lru_used_mb']} MB ({cc['mem_lru_evictions_delta']}) | "
            f"{cc['sqlite_patches_resident']} ({cc.get('sqlite_logical_mb', 0)} MB · {cc['sqlite_evictions_delta']}) |"
        )

    lines.extend(["", "## 2. Scaling by Cache Saturation Tier", ""])
    for pct in tiers:
        tier_rows = [s for s in stages if s["saturation_pct"] == pct]
        if not tier_rows:
            continue
        lines.append(f"### Saturation Tier {pct}%")
        for s in tier_rows:
            ol = s["overall_http_latency_ms"]
            tp = s["throughput"]
            cc = s["caches"]
            lines.append(
                f"- `C={s['concurrency']}`: `{tp['requests_per_sec']} req/s`, "
                f"mean `{ol['mean_ms']} ms`, `p95={ol['p95_ms']} ms`, `p99={ol['p99_ms']} ms`, "
                f"errors `{ol['total_errors']}`, DB evicts `+{cc['sqlite_evictions_delta']}`"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--vcpus", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--coadd-workers", type=int, default=0, help="0 = dynamic core formula")
    parser.add_argument("--in-memory-mb", type=int, default=1536)
    parser.add_argument("--sqlite-max-mb", type=int, default=4096)
    parser.add_argument("--auto-clamp-wsl", action="store_true")
    parser.add_argument("--users-per-tier", type=int, default=120)
    parser.add_argument("--users-per-stage", type=int, default=0, help="Deprecated alias for --users-per-tier")
    parser.add_argument("--stages", type=str, default="", help="Comma-separated concurrency list (default: 2,6,12,20,30,40)")
    parser.add_argument("--tiers", type=str, default="", help="Comma-separated saturation pcts (default: 50,85,100,125)")
    parser.add_argument("--no-reset-between-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.serve:
        run_stress_server(
            port=args.port,
            db_path=args.db,
            dataset_dir=args.dataset_dir,
            vcpus=args.vcpus,
            coadd_workers=args.coadd_workers,
            in_memory_mb=args.in_memory_mb,
            sqlite_max_mb=args.sqlite_max_mb,
        )
        return 0

    if not (args.dataset_dir / "sky_patches_catalog.json").exists():
        raise FileNotFoundError(
            f"Dataset missing in {args.dataset_dir}. Run generate_stress_dataset.py first."
        )

    avail_ram_mb, free_disk_mb = _probe_avail_ram_and_disk(args.db)
    in_memory_mb = args.in_memory_mb
    sqlite_max_mb = args.sqlite_max_mb
    if args.auto_clamp_wsl:
        in_memory_mb = min(in_memory_mb, max(256, avail_ram_mb - 900))
        sqlite_max_mb = min(sqlite_max_mb, max(512, int(free_disk_mb - 1200)))

    coadd_workers = (
        args.coadd_workers if args.coadd_workers > 0 else compute_dynamic_coadd_workers(args.vcpus)
    )

    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{args.db}{suffix}")
        if p.exists():
            p.unlink()

    patches = json.loads((args.dataset_dir / "sky_patches_catalog.json").read_text(encoding="utf-8"))
    action_lib = json.loads((args.dataset_dir / "action_library.json").read_text(encoding="utf-8"))
    archetypes = action_lib["archetypes"]
    action_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for act in action_lib["actions"]:
        action_buckets[act["endpoint"]].append(act)

    if args.smoke_test:
        concurrencies = [2, 4]
        tiers = [50, 100]
        users_per_tier = 2
    else:
        concurrencies = (
            [int(x.strip()) for x in args.stages.split(",") if x.strip()]
            if args.stages
            else list(DEFAULT_CONCURRENCIES)
        )
        tiers = (
            [int(x.strip()) for x in args.tiers.split(",") if x.strip()]
            if args.tiers
            else list(DEFAULT_SATURATION_TIERS)
        )
        users_per_tier = args.users_per_stage if args.users_per_stage > 0 else args.users_per_tier

    reset_between_stages = not args.no_reset_between_stages
    stages_cfg: list[dict[str, Any]] = []
    for c_idx, conc in enumerate(concurrencies, start=1):
        for t_idx, pct in enumerate(tiers, start=1):
            stages_cfg.append(
                {
                    "stage": c_idx,
                    "tier": t_idx,
                    "concurrency": conc,
                    "dataset_pct": pct,
                    "reset_before": reset_between_stages and t_idx == 1 and c_idx > 1,
                }
            )

    port = _pick_free_port(18090)
    base_url = f"http://127.0.0.1:{port}"
    log_path = BENCH_DIR / "stress_server.log"
    log_fh = log_path.open("w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            "--port",
            str(port),
            "--db",
            str(args.db),
            "--dataset-dir",
            str(args.dataset_dir),
            "--vcpus",
            str(args.vcpus),
            "--coadd-workers",
            str(coadd_workers),
            "--in-memory-mb",
            str(in_memory_mb),
            "--sqlite-max-mb",
            str(sqlite_max_mb),
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        import urllib.request

        ready = False
        for _ in range(120):
            if server_proc.poll() is not None:
                log_fh.flush()
                raise RuntimeError(
                    f"Stress server exited early:\n{log_path.read_text(encoding='utf-8')[-2000:]}"
                )
            try:
                with urllib.request.urlopen(f"{base_url}/internal/bench-cache-stats", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.15)
        if not ready:
            raise RuntimeError("Timed out waiting for stress server startup")

        sampler = ProcTelemetrySampler(server_proc.pid, args.db)
        sampler.start()

        async def _run_all() -> list[dict[str, Any]]:
            import httpx

            stage_results: list[dict[str, Any]] = []
            prev_pool = 0
            for st_spec in stages_cfg:
                if st_spec["reset_before"]:
                    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as rclient:
                        r = await rclient.post("/internal/bench-reset-cache")
                        r.raise_for_status()
                    prev_pool = 0
                    print(
                        f"Reset RAM & SQLite cache before concurrency stage C={st_spec['concurrency']}.",
                        flush=True,
                    )
                print(
                    f"Starting Stage {st_spec['stage']}.{st_spec['tier']}: C={st_spec['concurrency']}, "
                    f"saturation={st_spec['dataset_pct']}%, users={users_per_tier}...",
                    flush=True,
                )
                res, prev_pool = await run_stage(
                    st_spec,
                    base_url=base_url,
                    patches=patches,
                    prev_pool_size=prev_pool,
                    archetypes=archetypes,
                    action_buckets=action_buckets,
                    sampler=sampler,
                    users_per_stage=users_per_tier,
                    seed=args.seed,
                )
                print(
                    f"Done C={st_spec['concurrency']} @ {st_spec['dataset_pct']}% in {res['elapsed_s']}s | "
                    f"req/s={res['throughput']['requests_per_sec']} | "
                    f"mean={res['overall_http_latency_ms']['mean_ms']}ms | "
                    f"p95={res['overall_http_latency_ms']['p95_ms']}ms | "
                    f"err={res['overall_http_latency_ms']['total_errors']} | "
                    f"mb/patch={res['caches']['avg_sqlite_mb_per_patch']} | "
                    f"db_evict={res['caches']['sqlite_evictions_delta']}",
                    flush=True,
                )
                stage_results.append(res)
            return stage_results

        stage_results = asyncio.run(_run_all())
        sampler.stop()

        payload = {
            "config": {
                "vcpus": args.vcpus,
                "coadd_workers": coadd_workers,
                "in_memory_cache_mb": in_memory_mb,
                "sqlite_max_cache_mb": sqlite_max_mb,
                "dataset_patches": len(patches),
                "concurrencies": concurrencies,
                "tiers": tiers,
                "users_per_tier": users_per_tier,
                "reset_between_stages": reset_between_stages,
                "host_avail_ram_mb_at_start": avail_ram_mb,
                "host_free_disk_mb_at_start": round(free_disk_mb, 1),
            },
            "stages": stage_results,
        }
        if not args.smoke_test:
            RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            REPORT_PATH.write_text(build_stress_report(payload), encoding="utf-8")
            print(f"Saved stress test results to {RESULTS_PATH} and {REPORT_PATH}", flush=True)
        else:
            print("Stress smoke test passed cleanly.", flush=True)
        return 0
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
