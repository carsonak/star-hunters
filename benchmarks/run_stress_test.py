#!/usr/bin/env python3
"""5x5 Step-Load & Cache-Saturation Stress Test (C = 2..40 x ram_75..db_125).

Supports zero-eviction RAM tiers (ram_75, ram_97), disk-backed SQLite tiers
(db_75, db_97), and overflow eviction (db_125), plus --ram-ratio for sweeping
RAM:SQLite cache proportions and fast ram_75 snapshot pre-warming between
concurrency stages.

Generated logs/reports are stored under ``benchmarks/results/`` and SQLite
databases/datasets under ``benchmarks/data/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lib.host import (
    apply_cpu_affinity,
    compute_dynamic_coadd_workers,
    copy_db_group,
    percentiles,
    pick_free_port,
    probe_host_resources,
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

DATASET_DIR = DATA_DIR / "stress_dataset"
DB_PATH = DATA_DIR / "stress_cache.db"
RESULTS_PATH = RESULTS_DIR / "stress_test_results.json"
REPORT_PATH = RESULTS_DIR / "stress_test_report.md"

DEFAULT_CONCURRENCIES = [2, 6, 15, 25, 40]
DEFAULT_SATURATION_TIERS = ["ram_75", "ram_97", "db_75", "db_97", "db_125"]


def _parse_tier_spec(
    raw_tier: str,
    in_memory_mb: int,
    sqlite_max_mb: int,
    max_patches: int,
) -> dict[str, Any]:
    token = raw_tier.strip().lower()
    if token.startswith("ram_"):
        scope = "ram"
        pct = int(token.split("_", 1)[1].rstrip("%"))
    elif token.startswith("db_"):
        scope = "db"
        pct = int(token.split("_", 1)[1].rstrip("%"))
    else:
        scope = "db"
        pct = int(token.rstrip("%"))
        token = f"db_{pct}"
    base_mb = in_memory_mb if scope == "ram" else sqlite_max_mb
    pool_size = max(10, min(max_patches, int(round((base_mb * (pct / 100.0)) / 1.23))))
    return {
        "label": token,
        "scope": scope,
        "pct": pct,
        "pool_size": pool_size,
    }


def run_stress_server(
    port: int,
    db_path: Path,
    dataset_dir: Path,
    backend_dir: Path,
    vcpus: int,
    coadd_workers: int,
    in_memory_mb: int,
    sqlite_max_mb: int,
) -> None:
    """Launch an instrumented offline FastAPI stress-test server subprocess."""
    apply_cpu_affinity(vcpus)
    resolved_coadds = coadd_workers if coadd_workers > 0 else compute_dynamic_coadd_workers(vcpus)

    sys.path.insert(0, str(backend_dir))
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
    from app.services.spherex_coadd import build_target_wcs_header
    from app.services.spherex_fits import SpherexCutout, SpherexObservation
    from app.utils.astronomy import datetime_to_mjd

    catalog_raw = (dataset_dir / "sky_patches_catalog.json").read_text(encoding="utf-8")
    patches_data = json.loads(catalog_raw)
    by_grid_key = {
        p["grid_key"]: (
            int(p["template_id"]),
            float(p["flux_scale"]),
            p["gaia_rows"],
            p["simbad_rows"],
        )
        for p in patches_data
    }
    del patches_data

    with np.load(dataset_dir / "cutout_templates.npz") as templates:
        tpl_images = np.ascontiguousarray(templates["clean_images"], dtype=np.float32)
        tpl_vars = np.ascontiguousarray(templates["variances"], dtype=np.float32)
        tpl_flags = np.ascontiguousarray(templates["flags_masks"], dtype=np.uint32)
    tpl_wmaps = np.ascontiguousarray((1.0 / np.maximum(tpl_vars, 1e-4)), dtype=np.float32)
    for arr in (tpl_images, tpl_vars, tpl_flags, tpl_wmaps):
        arr.flags.writeable = False

    cache = SpatialSqliteCache(
        db_path=str(db_path),
        max_cache_bytes=int(sqlite_max_mb) * 1024 * 1024,
        max_memory_bytes=int(in_memory_mb) * 1024 * 1024,
    )
    seed_benchmark_ephemeris(
        cache,
        cc_mod.HORIZONS_FEATURED_BODIES,
        datetime_to_mjd(),
        source="jpl_horizons_stress",
    )
    stats_lock, stats = instrument_cache(cache, track_evictions=True)

    def dataset_cutout_fetcher(
        ra: float,
        dec: float,
        fov_arcmin: float = 5.0,
        max_cutouts: int | None = None,
        **_kwargs: Any,
    ) -> list[SpherexCutout]:
        gk, _, _ = compute_grid_key(ra, dec, fov_arcmin)
        pmeta = by_grid_key.get(gk)
        tid = pmeta[0] if pmeta else int(abs(round(ra * 10 + dec * 10))) % 32
        fscale = pmeta[1] if pmeta else 1.0
        mjds = (60500.0,) if (max_cutouts is not None and max_cutouts <= 6) else (60500.0, 60682.5)

        header = build_target_wcs_header(ra, dec, fov_arcmin / 60.0)
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        header["CRPIX1"] = 32.5
        header["CRPIX2"] = 32.5
        header["CDELT1"] = -(fov_arcmin / 60.0) / 64.0
        header["CDELT2"] = (fov_arcmin / 60.0) / 64.0

        cutouts: list[SpherexCutout] = []
        for ep_idx, mjd in enumerate(mjds):
            for det in range(1, 7):
                d_idx = det - 1
                img = tpl_images[tid, ep_idx, d_idx]
                if fscale != 1.0:
                    img = img * np.float32(fscale)
                cutouts.append(
                    SpherexCutout(
                        clean_image=img,
                        variance=tpl_vars[tid, ep_idx, d_idx],
                        weight_map=tpl_wmaps[tid, ep_idx, d_idx],
                        flags_mask=tpl_flags[tid, ep_idx, d_idx],
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
        return (list(pmeta[2]) if pmeta else []), "stress_gaia"

    async def dataset_simbad_fetch(
        ra_deg: float, dec_deg: float, *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict[str, Any]], str]:
        gk, _, _ = compute_grid_key(ra_deg, dec_deg, 5.0)
        pmeta = by_grid_key.get(gk)
        return (list(pmeta[3]) if pmeta else []), "stress_simbad"

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
            SpherexObservation(
                uri=f"stress://obs1_{ra:.2f}", bandpass="SPHEREx-D1", mjd_obs=60500.0
            ),
            SpherexObservation(
                uri=f"stress://obs2_{dec:.2f}", bandpass="SPHEREx-D2", mjd_obs=60682.5
            ),
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
    register_bench_stats_route(app, cache, stats_lock, stats, include_evictions=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _pick_patch_80_20(
    rng: random.Random,
    active_pool: list[dict[str, Any]],
    frontier_start_idx: int,
    frontier_deck: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drive tier saturation cleanly via ``frontier_deck`` while preserving hot-key reuse."""
    if frontier_deck and rng.random() < 0.85:
        return frontier_deck.pop()
    if rng.random() < 0.35 and frontier_start_idx < len(active_pool):
        return active_pool[rng.randint(frontier_start_idx, len(active_pool) - 1)]
    if rng.random() < 0.45:
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
    """Execute curated actions (zero timer sleeps) matching the chosen archetype."""
    target_actions = user_rng.randint(archetype["min_actions"], archetype["max_actions"])
    name = archetype["name"]
    actions_done = 0

    search_acts = action_buckets["/api/v1/catalogs/search"]
    e2s_acts = action_buckets["/api/v1/earth-to-sky"]
    fov_acts = action_buckets["/api/v1/solar-system/in-fov"]
    epoch_acts = action_buckets["/api/v1/catalogs/object-epochs"]
    most_acts = action_buckets["/api/v1/solar-system/most"]
    frame_modes = action_buckets["/api/v1/frames"]

    herd_patch = active_pool[min(len(active_pool) - 1, frontier_start_idx)]

    while actions_done < target_actions:
        if name == "rapid_globe_frustum_thrasher":
            for _ in range(user_rng.randint(8, 14)):
                act = user_rng.choice(fov_acts)
                await timed_get(client, act["endpoint"], act["params"], records, "in_fov")
                actions_done += 1
                if actions_done >= target_actions:
                    break
            act_s = user_rng.choice(search_acts)
            await timed_get(client, act_s["endpoint"], act_s["params"], records, "search")
            actions_done += 1
            continue

        if name in ("zenith_to_blinker", "capital_city_hopper"):
            e2s = user_rng.choice(e2s_acts)
            await timed_get(client, e2s["endpoint"], e2s["params"], records, "earth_to_sky")
            actions_done += 1
        elif name in ("search_blinker_spectra", "spectroscopy_lab_intensive"):
            sa = user_rng.choice(search_acts)
            await timed_get(client, sa["endpoint"], sa["params"], records, "search")
            actions_done += 1
        elif name == "solar_system_most_tracker":
            ma = user_rng.choice(most_acts)
            await timed_get(client, ma["endpoint"], ma["params"], records, "most")
            ea = user_rng.choice(epoch_acts)
            await timed_get(client, ea["endpoint"], ea["params"], records, "object_epochs")
            actions_done += 2

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
        fm = user_rng.choice(frame_modes)["params"]
        is_legacy = name == "legacy_float32_identity"
        precision = "float32" if is_legacy else fm["precision"]
        frame_headers = {"Accept-Encoding": "identity"} if is_legacy else None

        spec_prob = 0.85 if name == "spectroscopy_lab_intensive" else 0.45
        want_spectrum = user_rng.random() < spec_prob and (actions_done + 6) < target_actions
        actions_done += await fetch_patch_and_frames(
            client,
            ra,
            dec,
            records,
            catalogs_endpoint="catalogs",
            sky_patch_endpoint="sky_patch",
            frame_endpoint=f"frame:{precision}",
            spectrum_endpoint="spectrum",
            fetch_spectrum=want_spectrum,
            spectrum_label=patch["label"],
            precision=precision,
            frame_headers=frame_headers,
        )

        for _ in range(user_rng.randint(1, 3)):
            if actions_done >= target_actions:
                break
            fa = user_rng.choice(fov_acts)
            await timed_get(client, fa["endpoint"], fa["params"], records, "in_fov")
            actions_done += 1

    return actions_done


async def _prewarm_patch_pool(
    base_url: str,
    pool: list[dict[str, Any]],
    concurrency: int = 8,
) -> None:
    """Seed SQLite and RAM LRU with wire-ready float16 frames for ``pool``."""
    import httpx

    if not pool:
        return
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for p in pool:
        queue.put_nowait(p)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(60.0, connect=10.0),
        headers={"Accept-Encoding": "gzip"},
        limits=httpx.Limits(max_connections=max(32, concurrency * 6)),
    ) as client:

        async def _worker() -> None:
            while True:
                try:
                    patch = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                pr = await client.get(
                    "/api/v1/sky-patch",
                    params={
                        "ra": patch["ra"],
                        "dec": patch["dec"],
                        "fov": 5.0,
                        "live_catalogs": "false",
                    },
                )
                if pr.status_code == 200:
                    manifest = pr.json()
                    urls: list[tuple[str, dict[str, Any]]] = []
                    for ep in manifest.get("epochs", []):
                        if ep.get("detectors_data_url"):
                            urls.append((ep["detectors_data_url"], {"precision": "float16"}))
                        if ep.get("flags_mask_url"):
                            urls.append((ep["flags_mask_url"], {}))
                    if urls:
                        await asyncio.gather(*(client.get(u, params=p) for u, p in urls))

        await asyncio.gather(*(_worker() for _ in range(max(1, concurrency))))


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
    """Execute one Concurrency x Saturation cell and return aggregated stage metrics."""
    import httpx

    concurrency = int(stage_spec["concurrency"])
    pct = int(stage_spec["dataset_pct"])
    tier_label = str(stage_spec.get("tier_label", f"db_{pct}"))
    pool_size = max(10, min(len(patches), int(stage_spec["pool_size"])))
    active_pool = patches[:pool_size]
    frontier_start_idx = min(prev_pool_size, max(0, pool_size - 1))
    frontier_deck = (
        list(patches[prev_pool_size:pool_size]) if prev_pool_size < pool_size else []
    )
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
        limits=httpx.Limits(
            max_connections=max(64, concurrency * 8),
            max_keepalive_connections=max(32, concurrency * 4),
        ),
    ) as client:
        before_cache = (await client.get("/internal/bench-cache-stats")).json()
        sampler.reset_samples()
        t0 = time.perf_counter()

        async def _worker() -> None:
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
                flow_latencies_ms[arch["name"]].append((time.perf_counter() - ut0) * 1000.0)
                flow_counts[arch["name"]] += 1
                actions_executed += done

        await asyncio.gather(*(_worker() for _ in range(concurrency)))
        elapsed_s = time.perf_counter() - t0
        after_cache = (await client.get("/internal/bench-cache-stats")).json()

    proc_samples = sampler.snapshot()
    endpoint_summary = summarize_endpoints(records, include_extended=False)

    all_lats = [r["ms"] for r in records]
    all_p50, all_p95, all_p99 = percentiles(all_lats, (50.0, 95.0, 99.0))
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
            "p95_journey_ms": round(percentiles(lats, (95.0,))[0], 1),
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
            "tier_label": tier_label,
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
                "p50_ms": round(all_p50, 2),
                "p95_ms": round(all_p95, 2),
                "p99_ms": round(all_p99, 2),
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
    """Format the multi-stage stress test telemetry payload into a Markdown report."""
    cfg = payload["config"]
    stages = payload["stages"]
    concurrencies = cfg.get("concurrencies", [])
    tiers = cfg.get("tiers", [])
    ram_ratio = round(cfg["in_memory_cache_mb"] / max(1, cfg["sqlite_max_cache_mb"]), 3)
    matrix_hdr = (
        "| Concurrency | Tier | Pool | Reqs | Throughput (`req/s` · `act/s`) "
        "| Mean / p50 / p95 / p99 Latency (ms) | Errors | Avg/Peak CPU | Peak RSS "
        "| RAM Used (`Evicts`) | SQLite Patches (`Logical MB` · `Evicts`) |"
    )
    tier_hdr = (
        "| Stage | Tier | RAM LRU Hits | SQLite Hits | Cold Misses "
        "| RAM % / SQLite % / Combined Hit % | Cold Coadds |"
    )
    lines = [
        f"# {len(concurrencies)}×{len(tiers)} Step-Load & Cache-Saturation Stress Test Report",
        "",
        f"- **Target Configuration:** `{cfg['vcpus']}` vCPUs "
        f"(`{cfg['coadd_workers']}` dynamic coadd workers), "
        f"`{cfg['in_memory_cache_mb']} MB` RAM LRU, `{cfg['sqlite_max_cache_mb']} MB` SQLite Cap "
        f"(`RAM:SQLite = {ram_ratio}:1`, `prewarm={not cfg.get('no_prewarm', False)}`)",
        f"- **Dataset & Load:** `{cfg['dataset_patches']}` unique patches, "
        f"`{cfg['users_per_tier']}` spawned users per saturation tier "
        f"(`{cfg['users_per_tier'] * len(stages)}` total users, 0 timer sleeps)",
        f"- **Concurrencies:** `{concurrencies}` · **Saturation Tiers:** `{tiers}` "
        f"(`reset_between_stages={cfg.get('reset_between_stages', True)}`)",
        "",
        "## 1. Complete Concurrency × Saturation Matrix",
        "",
        matrix_hdr,
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in stages:
        ol = s["overall_http_latency_ms"]
        tp = s["throughput"]
        rs = s["resources"]
        cc = s["caches"]
        tlabel = s.get("tier_label", f"{s['saturation_pct']}%")
        sq_mb = cc.get("sqlite_logical_mb", 0)
        lines.append(
            f"| **C={s['concurrency']}** | **{tlabel}** | {s['active_patch_pool']} | "
            f"{s['total_http_requests']} | {tp['requests_per_sec']} · {tp['actions_per_sec']} | "
            f"**{ol['mean_ms']}** / {ol['p50_ms']} / {ol['p95_ms']} / {ol['p99_ms']} | "
            f"{ol['total_errors']} | {rs['avg_cpu_1vcpu_pct']}% / {rs['peak_cpu_1vcpu_pct']}% | "
            f"{rs['peak_rss_mb']} MB | {cc['mem_lru_used_mb']} MB "
            f"({cc['mem_lru_evictions_delta']}) | {cc['sqlite_patches_resident']} "
            f"({sq_mb} MB · {cc['sqlite_evictions_delta']}) |"
        )

    lines.extend(["", "## 2. Scaling by Cache Saturation Tier", ""])
    for tname in tiers:
        tier_rows = [
            s for s in stages if s.get("tier_label", str(s["saturation_pct"])) == str(tname)
        ]
        if not tier_rows:
            continue
        lines.append(f"### Tier `{tname}`")
        for s in tier_rows:
            ol = s["overall_http_latency_ms"]
            tp = s["throughput"]
            cc = s["caches"]
            lines.append(
                f"- `C={s['concurrency']}`: `{tp['requests_per_sec']} req/s` "
                f"(`{tp['egress_mbps']} Mbps`), mean `{ol['mean_ms']} ms`, "
                f"`p50={ol['p50_ms']} ms`, `p95={ol['p95_ms']} ms`, `p99={ol['p99_ms']} ms`, "
                f"errors `{ol['total_errors']}`, RAM evicts `+{cc['mem_lru_evictions_delta']}`, "
                f"DB evicts `+{cc['sqlite_evictions_delta']}`"
            )
        lines.append("")

    lines.extend([
        "## 3. Cache Tier Hit-Rate Breakdown (RAM LRU vs. SQLite vs. Cold Miss)",
        "",
        tier_hdr,
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for s in stages:
        to = s["caches"].get("tier_outcomes_delta", {})
        ram_h = sum(v for k, v in to.items() if k.endswith(":in_memory_lru"))
        sql_h = sum(v for k, v in to.items() if k.endswith(":sqlite_disk"))
        miss = sum(v for k, v in to.items() if k.endswith(":miss"))
        tot = max(1, ram_h + sql_h + miss)
        tlabel = s.get("tier_label", f"{s['saturation_pct']}%")
        lines.append(
            f"| C={s['concurrency']} | `{tlabel}` | {ram_h} | {sql_h} | {miss} | "
            f"{ram_h * 100.0 / tot:.1f}% / {sql_h * 100.0 / tot:.1f}% / "
            f"**{(ram_h + sql_h) * 100.0 / tot:.1f}%** | "
            f"{s['caches'].get('cold_coadds_delta', 0)} |"
        )

    last_tier = str(tiers[-1]) if tiers else "db_125"
    lines.extend(["", f"## 4. Endpoint Latency Breakdown at Max Saturation (`{last_tier}`)", ""])
    for s in [
        st for st in stages if st.get("tier_label", str(st["saturation_pct"])) == last_tier
    ]:
        tlabel = s.get("tier_label", f"{s['saturation_pct']}%")
        lines.extend([
            f"### `C={s['concurrency']}` @ `{tlabel}`",
            "",
            "| Endpoint | Count | Errors | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for ep, em in sorted(s.get("endpoints", {}).items(), key=lambda kv: -kv[1]["mean_ms"]):
            lines.append(
                f"| `{ep}` | {em['count']} | {em['errors']} | **{em['mean_ms']}** | "
                f"{em['p50_ms']} | {em['p95_ms']} | {em['p99_ms']} | {em['max_ms']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=None,
        help="Backend directory under test (defaults to STAR_HUNTERS_BACKEND_DIR or repo backend/)",
    )
    parser.add_argument("--vcpus", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--coadd-workers", type=int, default=0, help="0 = dynamic core formula")
    parser.add_argument("--in-memory-mb", type=int, default=192)
    parser.add_argument("--sqlite-max-mb", type=int, default=512)
    parser.add_argument(
        "--ram-ratio",
        type=float,
        default=0.0,
        help="If >0, set in-memory-mb = sqlite-max-mb * ram-ratio",
    )
    parser.add_argument(
        "--auto-clamp",
        action="store_true",
        help="Clamp --in-memory-mb and --sqlite-max-mb to safe probed host headroom",
    )
    parser.add_argument("--users-per-tier", type=int, default=50)
    parser.add_argument(
        "--users-per-stage",
        type=int,
        default=0,
        help="Deprecated alias for --users-per-tier",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated concurrency list (default: 2,6,15,25,40)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="",
        help="Comma-separated tiers (default: ram_75,ram_97,db_75,db_97,db_125)",
    )
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--no-reset-between-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    ensure_runtime_dirs()
    backend_dir = resolve_backend_dir(args.backend_dir)

    if args.serve:
        run_stress_server(
            port=args.port,
            db_path=args.db,
            dataset_dir=args.dataset_dir,
            backend_dir=backend_dir,
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

    host_probe = probe_host_resources(args.db)
    avail_ram_mb = int(host_probe["avail_ram_mb"])
    free_disk_mb = float(host_probe["effective_free_disk_mb"])
    sqlite_max_mb = args.sqlite_max_mb
    in_memory_mb = (
        max(32, int(round(sqlite_max_mb * args.ram_ratio)))
        if args.ram_ratio > 0.0
        else args.in_memory_mb
    )
    if args.smoke_test:
        in_memory_mb = 24
        sqlite_max_mb = 48
    elif args.auto_clamp:
        in_memory_mb = min(in_memory_mb, int(host_probe["max_safe_in_memory_mb"]))
        sqlite_max_mb = min(sqlite_max_mb, int(host_probe["max_safe_sqlite_mb"]))

    vcpus = min(args.vcpus, int(host_probe["cpu_cores_available"]))
    coadd_workers = (
        args.coadd_workers if args.coadd_workers > 0 else compute_dynamic_coadd_workers(vcpus)
    )

    raw_patches = json.loads(
        (args.dataset_dir / "sky_patches_catalog.json").read_text(encoding="utf-8")
    )
    patches = [
        {
            "ra": p["ra"],
            "dec": p["dec"],
            "label": p["label"],
            "grid_key": p["grid_key"],
            "regime": p["regime"],
        }
        for p in raw_patches
    ]
    del raw_patches

    action_lib = json.loads((args.dataset_dir / "action_library.json").read_text(encoding="utf-8"))
    archetypes = action_lib["archetypes"]
    action_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for act in action_lib["actions"]:
        action_buckets[act["endpoint"]].append(act)

    if args.smoke_test:
        concurrencies = [2, 4]
        raw_tiers = ["ram_75", "db_125"]
        users_per_tier = 2
    else:
        concurrencies = (
            [int(x.strip()) for x in args.stages.split(",") if x.strip()]
            if args.stages
            else list(DEFAULT_CONCURRENCIES)
        )
        raw_tiers = (
            [x.strip() for x in args.tiers.split(",") if x.strip()]
            if args.tiers
            else list(DEFAULT_SATURATION_TIERS)
        )
        users_per_tier = args.users_per_stage if args.users_per_stage > 0 else args.users_per_tier

    tier_specs = [
        _parse_tier_spec(t, in_memory_mb, sqlite_max_mb, len(patches)) for t in raw_tiers
    ]
    tiers = [ts["label"] for ts in tier_specs]
    reset_between_stages = not args.no_reset_between_stages
    stages_cfg: list[dict[str, Any]] = []
    for c_idx, conc in enumerate(concurrencies, start=1):
        for t_idx, tspec in enumerate(tier_specs, start=1):
            stages_cfg.append(
                {
                    "stage": c_idx,
                    "tier": t_idx,
                    "tier_label": tspec["label"],
                    "concurrency": conc,
                    "dataset_pct": tspec["pct"],
                    "pool_size": tspec["pool_size"],
                    "reset_before": reset_between_stages and t_idx == 1 and c_idx > 1,
                }
            )

    log_path = RESULTS_DIR / "stress_server.log"
    snapshot_db = args.db.with_name(f"{args.db.stem}_ram_prewarm.db")
    server_proc: subprocess.Popen[str] | None = None
    log_fh: Any = None
    sampler: ProcTelemetrySampler | None = None

    def _stop_server() -> None:
        nonlocal server_proc, log_fh, sampler
        if sampler is not None:
            sampler.stop()
            sampler = None
        if server_proc is not None:
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
            server_proc = None
        if log_fh is not None:
            log_fh.close()
            log_fh = None

    def _start_server(*, append_log: bool = False) -> str:
        nonlocal server_proc, log_fh, sampler
        import urllib.request

        port = pick_free_port(18090)
        base_url = f"http://127.0.0.1:{port}"
        log_fh = log_path.open("a" if append_log else "w", encoding="utf-8")
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
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

        ready = False
        for _ in range(200):
            if server_proc.poll() is not None:
                log_fh.flush()
                raise RuntimeError(
                    f"Stress server exited early:\n{log_path.read_text(encoding='utf-8')[-2000:]}"
                )
            try:
                stats_url = f"{base_url}/internal/bench-cache-stats"
                with urllib.request.urlopen(stats_url, timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.15)
        if not ready:
            raise RuntimeError("Timed out waiting for stress server startup")

        sampler = ProcTelemetrySampler(server_proc.pid, args.db)
        sampler.start()
        return base_url

    try:
        prewarm_count = 0 if args.no_prewarm else int(tier_specs[0]["pool_size"])
        unlink_db_group(args.db)
        unlink_db_group(snapshot_db)
        base_url = _start_server(append_log=False)

        if prewarm_count > 0:
            pw_t0 = time.perf_counter()
            tier0_lbl = tier_specs[0]["label"]
            print(
                f"Pre-warming initial {tier0_lbl} snapshot ({prewarm_count} patches)...",
                flush=True,
            )
            asyncio.run(_prewarm_patch_pool(base_url, patches[:prewarm_count]))
            _stop_server()
            copy_db_group(args.db, snapshot_db)
            base_url = _start_server(append_log=True)
            asyncio.run(_prewarm_patch_pool(base_url, patches[:prewarm_count], concurrency=12))
            print(
                f"Snapshot pre-warm ready in {time.perf_counter() - pw_t0:.2f}s.",
                flush=True,
            )

        async def _run_all() -> list[dict[str, Any]]:
            nonlocal base_url
            stage_results: list[dict[str, Any]] = []
            prev_pool = prewarm_count
            for st_spec in stages_cfg:
                if st_spec["reset_before"]:
                    _stop_server()
                    if prewarm_count > 0:
                        copy_db_group(snapshot_db, args.db)
                        base_url = _start_server(append_log=True)
                        await _prewarm_patch_pool(
                            base_url, patches[:prewarm_count], concurrency=12
                        )
                        prev_pool = prewarm_count
                        print(
                            f"Restored pre-warmed {tier_specs[0]['label']} snapshot before "
                            f"C={st_spec['concurrency']}.",
                            flush=True,
                        )
                    else:
                        unlink_db_group(args.db)
                        base_url = _start_server(append_log=True)
                        prev_pool = 0
                        print(
                            f"Reset SQLite DB before C={st_spec['concurrency']}.",
                            flush=True,
                        )
                print(
                    f"Starting Stage {st_spec['stage']}.{st_spec['tier']}: "
                    f"C={st_spec['concurrency']}, tier={st_spec['tier_label']} "
                    f"(pool={st_spec['pool_size']}), users={users_per_tier}...",
                    flush=True,
                )
                assert sampler is not None
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
                ol = res["overall_http_latency_ms"]
                cc = res["caches"]
                print(
                    f"Done C={st_spec['concurrency']} @ {st_spec['tier_label']} "
                    f"in {res['elapsed_s']}s | req/s={res['throughput']['requests_per_sec']} | "
                    f"mean={ol['mean_ms']}ms | p95={ol['p95_ms']}ms | "
                    f"err={ol['total_errors']} | mb/patch={cc['avg_sqlite_mb_per_patch']} | "
                    f"ram_ev={cc['mem_lru_evictions_delta']} | "
                    f"db_ev={cc['sqlite_evictions_delta']}",
                    flush=True,
                )
                stage_results.append(res)
            return stage_results

        stage_results = asyncio.run(_run_all())
        _stop_server()

        payload = {
            "config": {
                "vcpus": vcpus,
                "coadd_workers": coadd_workers,
                "in_memory_cache_mb": in_memory_mb,
                "sqlite_max_cache_mb": sqlite_max_mb,
                "ram_ratio": args.ram_ratio,
                "no_prewarm": args.no_prewarm,
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
            unlink_db_group(args.db)
            unlink_db_group(snapshot_db)
            print("Stress smoke test passed cleanly.", flush=True)
        return 0
    finally:
        _stop_server()


if __name__ == "__main__":
    raise SystemExit(main())
