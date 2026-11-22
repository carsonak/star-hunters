#!/usr/bin/env python3
"""Multi-configuration two-phase 20 connections/minute full-stack backend benchmark.

Key capabilities:
- 3-minute arrival windows (60 connections @ 20 conn/min over 180s per phase).
- High-variance user staying time across 3 explorer personas (60s-140s):
  Quick Explorer (~60-80s), Standard Researcher (~85-110s), Deep Hunter (~115-140s).
- Hard concurrency ceiling of 25 active virtual users (configurable via --max-concurrent).
- Dynamic archival coadd concurrency formula:
  reserves max(1, vcpus // 3) cores for I/O and asyncio (~2/3 allocated to CPU coadds).
- Physical-core-first Linux CPU affinity (os.sched_setaffinity) when available so
  constrained-vCPU runs pick distinct physical cores before SMT sibling threads.
- Request-weighted mean / p50 / p95 / p99 latencies across all and heavy endpoints.
- Machine-agnostic host resource probing via ``lib.host.probe_host_resources``.
- Runtime artifacts stored in git-ignored ``benchmarks/results/`` (JSON + logs) and
  ``benchmarks/data/`` (SQLite databases).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from lib.host import (
    apply_cpu_affinity,
    compute_dynamic_coadd_workers,
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
from lib.paths import DATA_DIR, RESULTS_DIR, ensure_runtime_dirs, resolve_backend_dir
from lib.telemetry import ProcTelemetrySampler

DB_PATH = DATA_DIR / "bench_cache.db"
RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"

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


def _load_60_cities(cities_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cities_csv.open("r", encoding="utf-8") as fh:
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


def _generate_cold_coords(
    seed: int,
    count: int,
    forbidden: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sample non-overlapping cold coordinates >= 2.2 deg apart with precomputed trig."""
    rng = random.Random(seed + 999)
    cos_min_sep = math.cos(math.radians(2.2))

    def _to_vec(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
        ra_r, dec_r = math.radians(ra_deg), math.radians(dec_deg)
        cos_d = math.cos(dec_r)
        return (cos_d * math.cos(ra_r), cos_d * math.sin(ra_r), math.sin(dec_r))

    existing_vecs = [_to_vec(ra, dec) for ra, dec in forbidden]
    chosen: list[tuple[float, float]] = []
    attempts = 0
    while len(chosen) < count and attempts < 80_000:
        attempts += 1
        ra = round(rng.uniform(0.0, 359.8), 4)
        dec = round(rng.uniform(-75.0, 75.0), 4)
        vx, vy, vz = _to_vec(ra, dec)
        if any(vx * ex + vy * ey + vz * ez > cos_min_sep for ex, ey, ez in existing_vecs):
            continue
        existing_vecs.append((vx, vy, vz))
        chosen.append((ra, dec))
    return chosen


def _build_arrival_offsets(
    rng: random.Random,
    total_sessions: int = 60,
    duration_s: float = 180.0,
) -> list[float]:
    """Schedule connections in bursty waves at 20 conn/min over ``duration_s``."""
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


def run_bench_server(
    port: int,
    db_path: Path,
    backend_dir: Path,
    vcpus: int,
    coadd_workers: int,
    in_memory_mb: int,
    sqlite_max_mb: int,
) -> None:
    """Launch an instrumented offline FastAPI benchmark server subprocess."""
    import os

    apply_cpu_affinity(vcpus)
    resolved_coadd_workers = (
        coadd_workers if coadd_workers > 0 else compute_dynamic_coadd_workers(vcpus)
    )

    sys.path.insert(0, str(backend_dir))
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
    from app.services import spherex_ingest as si_mod
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
    seed_benchmark_ephemeris(
        cache,
        cc_mod.HORIZONS_FEATURED_BODIES,
        datetime_to_mjd(),
        source="jpl_horizons_bench",
    )
    stats_lock, stats = instrument_cache(cache, track_evictions=False)

    def mock_cutout_fetcher(
        ra: float,
        dec: float,
        fov_arcmin: float = 5.0,
        max_cutouts: int | None = None,
        **_kwargs: Any,
    ) -> list[Any]:
        time.sleep(0.03)
        mjds = (60500.0,) if (max_cutouts is not None and max_cutouts <= 6) else (60500.0, 60682.5)
        cutouts = [
            generate_synthetic_spherex_cutout(
                ra=ra,
                dec=dec,
                fov_arcmin=fov_arcmin,
                bandpass=f"SPHEREx-D{det}",
                shape=(64, 64),
                mjd_obs=mjd,
            )
            for mjd in mjds
            for det in range(1, 7)
        ]
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

    ccp_mod.fetch_simbad_objects = no_live_fetch  # type: ignore[assignment]
    ccp_mod.fetch_gaia_dr3_objects = no_live_fetch  # type: ignore[assignment]
    bs_mod.fetch_most_spherex_matches = no_live_most  # type: ignore[assignment]
    oe_mod.query_observations = lambda *_a, **_kw: []  # type: ignore[assignment]
    si_mod.query_observations = lambda *_a, **_kw: []  # type: ignore[assignment]

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
    register_bench_stats_route(app, cache, stats_lock, stats, include_evictions=False)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _compute_weighted_latency_summary(
    endpoint_summary: dict[str, Any],
    *,
    prefixes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    selected = [
        v for k, v in endpoint_summary.items() if prefixes is None or k.startswith(prefixes)
    ]
    tot_count = sum(v["count"] for v in selected)
    if tot_count <= 0:
        return {
            "mean_ms": 0.0,
            "weighted_p50_ms": 0.0,
            "weighted_p95_ms": 0.0,
            "weighted_p99_ms": 0.0,
        }
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
    """Execute one 20-conn/min arrival-wave phase and return telemetry + latency summary."""
    import httpx

    rng = random.Random(seed)
    arrival_offsets = _build_arrival_offsets(
        rng, total_sessions=total_sessions, duration_s=duration_s
    )
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
                    remaining = (slot_index * slot_target) - (time.perf_counter() - user_t0)
                    if remaining > 0.02:
                        await asyncio.sleep(remaining)

                active_sessions += 1
                max_concurrent = max(max_concurrent, active_sessions)
                concurrency_samples.append(active_sessions)
                try:
                    # Cluster 1: Globe Search & Coordinate / Epoch / MOST Lookup
                    for q in user_rng.sample(SEARCH_QUERIES, k=persona["searches"]):
                        await timed_get(
                            client,
                            "/api/v1/catalogs/search",
                            {"q": q, "limit": 12},
                            records,
                            "search",
                        )
                    if persona["check_object_epochs"]:
                        slug0, ra0, dec0 = user_rng.choice(hot_pool)
                        await timed_get(
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
                        await timed_get(
                            client,
                            "/api/v1/solar-system/most",
                            {"obj_name": user_rng.choice(MOST_TARGETS)},
                            records,
                            "most",
                        )
                    await _pace_until(1)

                    # Cluster 2: Globe Panning & Zooming (FOV sweeps)
                    for _ in range(persona["fov_sweeps"]):
                        await timed_get(
                            client,
                            "/api/v1/solar-system/in-fov",
                            {
                                "ra": round(user_rng.uniform(0.0, 359.5), 3),
                                "dec": round(user_rng.uniform(-75.0, 75.0), 3),
                                "fov": user_rng.choice([5.0, 15.0, 45.0]),
                            },
                            records,
                            "in_fov",
                        )
                    await _pace_until(2)

                    # Cluster 3: Cold Cache Navigation
                    for c_offset in range(persona["cold_patches"]):
                        c_ra, c_dec = cold_coords[(user_idx * 2 + c_offset) % len(cold_coords)]
                        await fetch_patch_and_frames(
                            client,
                            c_ra,
                            c_dec,
                            records,
                            catalogs_endpoint="catalogs:cold",
                            sky_patch_endpoint="sky_patch:cold",
                            frame_endpoint="frame:cold",
                            spectrum_endpoint="spectrum:cold",
                            fetch_spectrum=(user_rng.random() < persona["spectra_prob"]),
                            track_cached=True,
                        )
                    await _pace_until(3)

                    # Cluster 4: Purposeful Hot Navigation (first half)
                    hot_picks = user_rng.sample(hot_pool, k=persona["hot_patches"])
                    half = max(1, len(hot_picks) // 2)
                    for h_idx, (_slug, h_ra, h_dec) in enumerate(hot_picks[:half]):
                        want_spec = h_idx == 0 and user_rng.random() < persona["spectra_prob"]
                        await fetch_patch_and_frames(
                            client,
                            h_ra,
                            h_dec,
                            records,
                            catalogs_endpoint="catalogs:hot",
                            sky_patch_endpoint="sky_patch:hot",
                            frame_endpoint="frame:hot",
                            spectrum_endpoint="spectrum:hot",
                            fetch_spectrum=want_spec,
                            track_cached=True,
                        )
                    await _pace_until(4)

                    # Cluster 5: Remaining Hot Navigation
                    for h_idx, (_slug, h_ra, h_dec) in enumerate(hot_picks[half:]):
                        want_spec = h_idx == 0 and user_rng.random() < persona["spectra_prob"]
                        await fetch_patch_and_frames(
                            client,
                            h_ra,
                            h_dec,
                            records,
                            catalogs_endpoint="catalogs:hot",
                            sky_patch_endpoint="sky_patch:hot",
                            frame_endpoint="frame:hot",
                            spectrum_endpoint="spectrum:hot",
                            fetch_spectrum=want_spec,
                            track_cached=True,
                        )
                    await _pace_until(5)

                    # Cluster 6: Earth-to-Sky City Zenith Navigation
                    picked_cities = user_rng.sample(cities_60, k=3)
                    zenith_targets: list[tuple[float, float]] = []
                    for city in picked_cities:
                        er = await timed_get(
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
                        await fetch_patch_and_frames(
                            client,
                            z_ra,
                            z_dec,
                            records,
                            catalogs_endpoint="catalogs:city_zenith",
                            sky_patch_endpoint="sky_patch:city_zenith",
                            frame_endpoint="frame:city_zenith",
                            spectrum_endpoint="spectrum:city_zenith",
                            fetch_spectrum=(user_rng.random() < persona["spectra_prob"]),
                            track_cached=True,
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

    proc_samples = sampler.snapshot()
    delta_tiers = {
        k: after_cache["tier_outcomes"].get(k, 0) - before_cache["tier_outcomes"].get(k, 0)
        for k in sorted(set(before_cache["tier_outcomes"]) | set(after_cache["tier_outcomes"]))
    }
    delta_writes = {
        k: after_cache["cold_writes"].get(k, 0) - before_cache["cold_writes"].get(k, 0)
        for k in ("sky_patch", "spectrum", "zenith")
    }

    endpoint_summary = summarize_endpoints(records, include_extended=True)
    overall_latency_ms = _compute_weighted_latency_summary(endpoint_summary)
    heavy_endpoints_latency_ms = _compute_weighted_latency_summary(
        endpoint_summary,
        prefixes=("sky_patch:", "spectrum:", "earth_to_sky", "object_epochs"),
    )

    cpu_vals = [s["cpu_pct_total"] for s in proc_samples]
    rss_vals = [s["rss_mb"] for s in proc_samples]
    hwm_vals = [s["hwm_mb"] for s in proc_samples]
    db_vals = [s["db_disk_mb"] for s in proc_samples]
    stay_p50, stay_p95 = percentiles(stay_times_s, (50.0, 95.0))
    (cpu_p95,) = percentiles(cpu_vals, (95.0,))

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
            "p50": round(stay_p50, 2) if stay_times_s else 0.0,
            "mean": round(statistics.mean(stay_times_s), 2) if stay_times_s else 0.0,
            "p95": round(stay_p95, 2) if stay_times_s else 0.0,
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
            "p95_pct_1vcpu_norm": round(cpu_p95, 1) if cpu_vals else 0.0,
            "peak_pct_1vcpu_norm": round(max(cpu_vals), 1) if cpu_vals else 0.0,
            "avg_pct_allocated_vcpus": round(statistics.mean(cpu_vals) / norm_cores, 1)
            if cpu_vals
            else 0.0,
            "peak_pct_allocated_vcpus": round(max(cpu_vals) / norm_cores, 1) if cpu_vals else 0.0,
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
    backend_dir: Path,
    total_sessions: int,
    duration_s: float,
    max_concurrent_users: int,
    smoke_test: bool,
) -> dict[str, Any]:
    """Run Run 1 (Cold) and Run 2 (Prewarmed) against one resource configuration."""
    unlink_db_group(db_path)

    port = pick_free_port(18080)
    base_url = f"http://127.0.0.1:{port}"
    cfg_name = cfg["name"]
    vcpus = int(cfg["vcpus"])
    coadd_workers = int(cfg.get("coadd_workers") or compute_dynamic_coadd_workers(vcpus))
    in_memory_mb = int(cfg["in_memory_cache_mb"])
    sqlite_max_mb = int(cfg["sqlite_max_cache_mb"])

    server_log_path = RESULTS_DIR / f"bench_server_{cfg_name}.log"
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
            "--backend-dir",
            str(backend_dir),
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

        sampler = ProcTelemetrySampler(server_proc.pid, db_path)
        sampler.start()

        async def _execute_pair() -> tuple[dict[str, Any], dict[str, Any]]:
            print(
                f"[{cfg_name}] Run 1 (Cold Start, seed=42, {total_sessions} sessions over "
                f"{duration_s}s, vcpus={vcpus} cpus={select_physical_first_cpus(vcpus)}, "
                f"coadd_workers={coadd_workers}, ram_lru={in_memory_mb}MB, "
                f"sqlite={sqlite_max_mb}MB)...",
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
                f"[{cfg_name}] Run 1 done in {r1['elapsed_s']}s | "
                f"mean_ms={r1['overall_latency_ms']['mean_ms']}ms | "
                f"w_p50={r1['overall_latency_ms']['weighted_p50_ms']}ms | "
                f"peak_rss={r1['server_ram']['peak_rss_mb']}MB | "
                f"lru={r1['application_caches']['mem_lru_used_mb_after']}MB | "
                f"db={r1['disk_and_sqlite']['end_db_mb']}MB",
                flush=True,
            )

            await asyncio.sleep(2.0)
            print(
                f"[{cfg_name}] Run 2 (Prewarmed, seed=1337, {total_sessions} sessions over "
                f"{duration_s}s)...",
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
                f"[{cfg_name}] Run 2 done in {r2['elapsed_s']}s | "
                f"mean_ms={r2['overall_latency_ms']['mean_ms']}ms | "
                f"w_p50={r2['overall_latency_ms']['weighted_p50_ms']}ms | "
                f"peak_rss={r2['server_ram']['peak_rss_mb']}MB | "
                f"lru={r2['application_caches']['mem_lru_used_mb_after']}MB | "
                f"db={r2['disk_and_sqlite']['end_db_mb']}MB",
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
                f"Invalid --configs entry '{item}'; "
                "expected name:vcpus:coadd_workers:in_mem_mb:sqlite_mb"
            )
        name, vc, cw, mem_mb, db_mb = parts
        vcpus_int = int(vc)
        cw_int = int(cw) if int(cw) > 0 else compute_dynamic_coadd_workers(vcpus_int)
        parsed.append(
            {
                "name": name,
                "fly_equivalent": (
                    f"{vcpus_int} vCPU ({cw_int} coadd) / {mem_mb}MB LRU / {db_mb}MB SQLite"
                ),
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
    parser.add_argument(
        "--matrix", action="store_true", help="Run the auto-probed multi-config matrix"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help=(
            "Comma-separated name:vcpus:coadd_workers:in_mem_mb:sqlite_mb configs "
            "(0 coadd_workers = dynamic formula)"
        ),
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=None,
        help="Backend directory under test (defaults to STAR_HUNTERS_BACKEND_DIR or repo backend/)",
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

    ensure_runtime_dirs()
    backend_dir = resolve_backend_dir(args.backend_dir)

    if args.serve:
        run_bench_server(
            port=args.port,
            db_path=args.db,
            backend_dir=backend_dir,
            vcpus=args.vcpus,
            coadd_workers=args.coadd_workers,
            in_memory_mb=args.in_memory_mb or 512,
            sqlite_max_mb=args.sqlite_max_mb or 750,
        )
        return 0

    host_probe = probe_host_resources(args.db)
    if args.probe_resources:
        print(json.dumps(host_probe, indent=2), flush=True)
        return 0

    sys.path.insert(0, str(backend_dir))
    from app.models.contracts import SHOWCASE_TARGETS

    cities_csv = backend_dir / "app" / "models" / "world_capital_cities.csv"
    hot_pool = [(slug, float(m["ra"]), float(m["dec"])) for slug, m in SHOWCASE_TARGETS.items()]
    hot_pool.extend(HOT_EXTRA_COORDS)
    cities_60 = _load_60_cities(cities_csv)

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
        configs = list(host_probe["recommended_matrix"])
    else:
        in_mem = args.in_memory_mb or host_probe["max_safe_in_memory_mb"]
        sq_max = args.sqlite_max_mb or host_probe["max_safe_sqlite_mb"]
        vcpus = min(args.vcpus, int(host_probe["cpu_cores_available"]))
        cw = args.coadd_workers or compute_dynamic_coadd_workers(vcpus)
        configs = [
            {
                "name": f"cfg_{vcpus}vcpu_{in_mem}m_ram",
                "fly_equivalent": (
                    f"shared-cpu-{vcpus}x ({cw} coadd, {in_mem}MB LRU / {sq_max}MB SQLite)"
                ),
                "vcpus": vcpus,
                "coadd_workers": cw,
                "pinned_cpus": select_physical_first_cpus(vcpus),
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
            backend_dir=backend_dir,
            total_sessions=total_sessions,
            duration_s=duration_s,
            max_concurrent_users=args.max_concurrent,
            smoke_test=args.smoke_test,
        )
        config_runs.append(res)

    primary = config_runs[-1]
    output_payload = {
        "host_resources_at_start": host_probe,
        "configuration_runs": config_runs,
        "backend_run_1": primary["backend_run_1"],
        "backend_run_2": primary["backend_run_2"],
    }
    if not args.smoke_test:
        RESULTS_PATH.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(f"Saved multi-config benchmark results to {RESULTS_PATH}", flush=True)
    else:
        unlink_db_group(args.db)
        print("Smoke test completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
