"""Shared cache instrumentation, ephemeris seeding, and FastAPI stats endpoints."""

from __future__ import annotations

import functools
import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def seed_benchmark_ephemeris(
    cache: Any,
    bodies: Sequence[dict[str, Any]],
    now_mjd: float,
    *,
    source: str,
) -> None:
    """Populate synthetic deterministic ephemeris points for offline Solar System queries."""
    ephem_points: list[dict[str, Any]] = []
    for idx, spec in enumerate(bodies):
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
                    "source": source,
                }
            )
    cache.upsert_ephemeris_points(ephem_points)


def instrument_cache(
    cache: Any,
    *,
    track_evictions: bool = False,
) -> tuple[threading.Lock, dict[str, Any]]:
    """Wrap ``SpatialSqliteCache`` read/write/peek methods to record tier hit/miss telemetry."""
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

    def tracked_mem_get(key: str) -> bytes | None:
        res = orig_mem_get(key)
        with stats_lock:
            if res is not None:
                stats["mem_lru_raw_hits"] += 1
            else:
                stats["mem_lru_raw_misses"] += 1
        return res

    cache._mem_cache.get = tracked_mem_get

    if track_evictions:
        orig_mem_put = cache._mem_cache.put

        def tracked_mem_put(key: str, value: bytes) -> None:
            before_entries = cache._mem_cache.entry_count
            existed = key in cache._mem_cache._entries
            orig_mem_put(key, value)
            after_entries = cache._mem_cache.entry_count
            evicted = max(0, (before_entries + (0 if existed else 1)) - after_entries)
            if evicted > 0:
                with stats_lock:
                    stats["mem_lru_evictions"] += evicted

        cache._mem_cache.put = tracked_mem_put

        orig_prune = cache._prune_if_needed

        def tracked_prune(conn: Any, preserve_grid_key: str | None = None) -> None:
            before_cnt = int(conn.execute("SELECT COUNT(*) FROM sky_patches").fetchone()[0])
            orig_prune(conn, preserve_grid_key=preserve_grid_key)
            after_cnt = int(conn.execute("SELECT COUNT(*) FROM sky_patches").fetchone()[0])
            if before_cnt > after_cnt:
                with stats_lock:
                    stats["sqlite_evictions"] += before_cnt - after_cnt

        cache._prune_if_needed = tracked_prune

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
    orig_peek_manifest = cache.peek_manifest_memory
    orig_peek_catalogs = cache.peek_catalogs_memory
    orig_peek_spectrum = cache.peek_spectrum_memory
    orig_peek_frame_gzip = cache.peek_frame_gzip_memory

    @functools.wraps(orig_get_manifest)
    def wrap_get_manifest(grid_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"manifest:{grid_key}")
        res = orig_get_manifest(grid_key, *args, **kwargs)
        _record_tier("manifest", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_catalogs)
    def wrap_get_catalogs(grid_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"catalogs:{grid_key}")
        res = orig_get_catalogs(grid_key, *args, **kwargs)
        _record_tier("catalogs", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_frame_gzip)
    def wrap_get_frame_gzip(
        grid_key: str, filename: str, precision: str = "float32", *args: Any, **kwargs: Any
    ) -> Any:
        if precision == "float16" and not filename.endswith("_flags.bin"):
            return orig_get_frame_gzip(grid_key, filename, precision=precision, *args, **kwargs)
        dtype_hdr = "uint8" if filename.endswith("_flags.bin") else "float32"
        in_ram = _in_mem_peek(f"frame_gz:{grid_key}:{filename}:{dtype_hdr}") or _in_mem_peek(
            f"frame_raw:{grid_key}:{filename}"
        )
        res = orig_get_frame_gzip(grid_key, filename, precision=precision, *args, **kwargs)
        _record_tier("frame_gzip", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_frame)
    def wrap_get_frame(grid_key: str, filename: str, *args: Any, **kwargs: Any) -> bytes | None:
        in_ram = _in_mem_peek(f"frame_raw:{grid_key}:{filename}")
        res = orig_get_frame(grid_key, filename, *args, **kwargs)
        _record_tier("frame_raw", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_spectrum)
    def wrap_get_spectrum(spectrum_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"spectrum:{spectrum_key}")
        res = orig_get_spectrum(spectrum_key, *args, **kwargs)
        _record_tier("spectrum", in_ram, res is not None)
        return res

    @functools.wraps(orig_get_zenith)
    def wrap_get_zenith(cache_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        in_ram = _in_mem_peek(f"zenith:{cache_key}")
        res = orig_get_zenith(cache_key, *args, **kwargs)
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

    @functools.wraps(orig_peek_manifest)
    def wrap_peek_manifest(grid_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        res = orig_peek_manifest(grid_key, *args, **kwargs)
        if res is not None:
            _record_tier("manifest", True, True)
        return res

    @functools.wraps(orig_peek_catalogs)
    def wrap_peek_catalogs(grid_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        res = orig_peek_catalogs(grid_key, *args, **kwargs)
        if res is not None:
            _record_tier("catalogs", True, True)
        return res

    @functools.wraps(orig_peek_spectrum)
    def wrap_peek_spectrum(spectrum_key: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        res = orig_peek_spectrum(spectrum_key, *args, **kwargs)
        if res is not None:
            _record_tier("spectrum", True, True)
        return res

    @functools.wraps(orig_peek_frame_gzip)
    def wrap_peek_frame_gzip(
        grid_key: str, filename: str, precision: str = "float32", *args: Any, **kwargs: Any
    ) -> Any:
        res = orig_peek_frame_gzip(grid_key, filename, precision=precision, *args, **kwargs)
        if res is not None:
            _record_tier("frame_gzip", True, True)
        return res

    cache.get_manifest = wrap_get_manifest
    cache.get_catalogs = wrap_get_catalogs
    cache.get_frame_gzip = wrap_get_frame_gzip
    cache.get_frame = wrap_get_frame
    cache.get_spectrum = wrap_get_spectrum
    cache.get_zenith_response = wrap_get_zenith
    cache.peek_manifest_memory = wrap_peek_manifest
    cache.peek_catalogs_memory = wrap_peek_catalogs
    cache.peek_spectrum_memory = wrap_peek_spectrum
    cache.peek_frame_gzip_memory = wrap_peek_frame_gzip
    cache.put_sky_patch = wrap_put_patch
    cache.put_spectrum = wrap_put_spec
    cache.put_zenith_response = wrap_put_zenith

    return stats_lock, stats


def register_bench_stats_route(
    app: Any,
    cache: Any,
    stats_lock: threading.Lock,
    stats: dict[str, Any],
    *,
    include_evictions: bool = False,
) -> None:
    """Register ``GET /internal/bench-cache-stats`` ahead of catch-all SPA routes."""

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
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(LENGTH(compressed_blob)), 0) AS comp_bytes, "
                "COALESCE(SUM(raw_byte_size), 0) AS raw_bytes FROM binary_frames"
            ).fetchone()
            spec_row = conn.execute("SELECT COUNT(*) AS cnt FROM spectra_cache").fetchone()
            zen_row = conn.execute("SELECT COUNT(*) AS cnt FROM capital_zenith_cache").fetchone()

        payload: dict[str, Any] = {
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
        if include_evictions:
            payload["mem_lru_evictions"] = mem_evict
            payload["sqlite_evictions"] = sq_evict
        return payload

    app.router.routes.insert(0, app.router.routes.pop())
