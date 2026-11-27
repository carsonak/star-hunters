"""Tests for Phase 3/4 lifespan pooling, SQLite deduplication, preset_builder, and contracts."""

from __future__ import annotations

import asyncio
import copy
import gzip
import importlib.util
import json
import os
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import CachedStaticFiles, create_app
from app.models.contracts import (
    BINARY_DETECTOR_FLOAT16_BYTES,
    SHOWCASE_TARGETS,
    TRACKING_SQUARE_COLORS,
    convert_float32_bytes_to_float16_bytes,
    validate_catalogs_response,
    validate_sky_patch_manifest,
)
from app.services import synthetic_patch
from app.services.backend_service import StarHuntersBackendService
from app.services.catalog_clients import HORIZONS_FEATURED_BODIES
from app.services.celestial_catalog_pipeline import AutomatedCelestialPipeline
from app.services.preset_builder import (
    PresetBundle,
    build_live_preset_bundle,
    build_synthetic_preset_bundle,
    generate_all_synthetic_presets,
    prewarm_cache_with_service,
    seed_ephemeris_for_bodies,
    seed_most_intersections,
    seed_showcase_regions,
    validate_preset_bundle,
    validate_written_presets,
    write_preset_bundle,
    write_summary_json,
)
from app.services.spatial_cache import SpatialSqliteCache, compute_grid_key
from app.services.synthetic_patch import (
    build_sky_patch_manifest_payload,
    generate_synthetic_epoch_frames,
)


class LifespanCacheAndPresetsTests(unittest.TestCase):
    """Verify ASGI HTTP pooling, SQLite JSON deduplication, preset_builder, and contract parity."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.db_path = str(self.tmp_path / "phase34_test.db")
        self.cache = SpatialSqliteCache(db_path=self.db_path, max_cache_bytes=64 * 1024 * 1024)

    def tearDown(self) -> None:
        self.cache.close()
        self._tmpdir.cleanup()

    def test_fastapi_lifespan_manages_pooled_http_client_and_closes_cache(self) -> None:
        service = StarHuntersBackendService(cache=self.cache)
        self.assertIsNone(service.catalog_pipeline.http_client)

        managed_client: httpx.AsyncClient | None = None
        with TestClient(create_app(service)) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            managed_client = service.catalog_pipeline.http_client
            self.assertIsInstance(managed_client, httpx.AsyncClient)
            assert managed_client is not None
            self.assertFalse(managed_client.is_closed)

        self.assertIsNotNone(managed_client)
        assert managed_client is not None
        self.assertTrue(managed_client.is_closed)
        self.assertIsNone(service.catalog_pipeline.http_client)
        self.assertEqual(self.cache._connections, [])

    def test_fastapi_lifespan_preserves_injected_http_client(self) -> None:
        async def _run() -> None:
            transport = httpx.MockTransport(lambda _r: httpx.Response(200))
            injected = httpx.AsyncClient(transport=transport)
            pipeline = AutomatedCelestialPipeline(cache=self.cache, http_client=injected)
            service = StarHuntersBackendService(cache=self.cache, catalog_pipeline=pipeline)
            app = create_app(service)
            async with app.router.lifespan_context(app):
                self.assertIs(service.catalog_pipeline.http_client, injected)
                self.assertFalse(injected.is_closed)
            self.assertIs(service.catalog_pipeline.http_client, injected)
            self.assertFalse(injected.is_closed)
            await injected.aclose()

        asyncio.run(_run())

    def test_pipeline_creates_and_closes_transient_client_outside_lifespan(self) -> None:
        pipeline = AutomatedCelestialPipeline(cache=self.cache, http_client=None)
        observed_clients: list[httpx.AsyncClient] = []

        async def _fake_simbad(
            *,
            ra_deg: float,
            dec_deg: float,
            radius_deg: float,
            client: httpx.AsyncClient | None = None,
            parallel_mirrors: bool = False,
        ) -> tuple[list[dict[str, object]], str]:
            assert client is not None
            observed_clients.append(client)
            self.assertFalse(client.is_closed)
            return [], "simbad_mock"

        async def _fake_gaia(
            *,
            ra_deg: float,
            dec_deg: float,
            radius_deg: float,
            client: httpx.AsyncClient | None = None,
            parallel_mirrors: bool = False,
        ) -> tuple[list[dict[str, object]], str]:
            assert client is not None
            observed_clients.append(client)
            self.assertFalse(client.is_closed)
            return [], "gaia_mock"

        wcs = {
            "width": 256,
            "height": 256,
            "crval": [133.8, -7.25],
            "crpix": [128.0, 128.0],
            "cdelt": [-0.0003255, 0.0003255],
        }
        epochs = [{"epoch_id": "pass_1", "mjd": 60810.5}]

        with (
            patch(
                "app.services.celestial_catalog_pipeline.fetch_simbad_objects",
                side_effect=_fake_simbad,
            ),
            patch(
                "app.services.celestial_catalog_pipeline.fetch_gaia_dr3_objects",
                side_effect=_fake_gaia,
            ),
        ):
            asyncio.run(pipeline.resolve_patch_objects(133.8, -7.25, 5.0, wcs, epochs))

        self.assertEqual(len(observed_clients), 2)
        self.assertIs(observed_clients[0], observed_clients[1])
        self.assertTrue(observed_clients[0].is_closed)
        self.assertIsNone(pipeline.http_client)

    def test_sqlite_manifest_and_catalogs_deduplicated_round_trip(self) -> None:
        meta = SHOWCASE_TARGETS["wise_0855"]
        manifest = build_sky_patch_manifest_payload(
            ra=meta["ra"],
            dec=meta["dec"],
            fov_arcmin=5.0,
            slug="wise_0855",
        )
        original_overlays = copy.deepcopy(manifest["catalog_overlays"])
        frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
        grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)

        self.cache.put_sky_patch(
            grid_key=grid_key,
            ra_center=ra_q,
            dec_center=dec_q,
            fov_arcmin=5.0,
            manifest=manifest,
            frames=frames,
        )
        # Caller manifest must not be mutated
        self.assertIn("catalog_overlays", manifest)

        with self.cache.connection() as conn:
            row = conn.execute(
                "SELECT manifest_json, catalogs_json FROM sky_patches WHERE grid_key = ?",
                (grid_key,),
            ).fetchone()
        self.assertIsNotNone(row)
        stored_manifest = json.loads(row["manifest_json"])
        stored_catalogs = json.loads(row["catalogs_json"])
        self.assertNotIn("catalog_overlays", stored_manifest)
        self.assertIn("catalog_overlays", stored_catalogs)

        hydrated_manifest = self.cache.get_manifest(grid_key)
        self.assertIsNotNone(hydrated_manifest)
        assert hydrated_manifest is not None
        self.assertTrue(hydrated_manifest["query"]["cached"])
        self.assertEqual(hydrated_manifest["catalog_overlays"], original_overlays)
        validate_sky_patch_manifest(hydrated_manifest)

        cached_catalogs = self.cache.get_catalogs(grid_key)
        self.assertIsNotNone(cached_catalogs)
        assert cached_catalogs is not None
        self.assertTrue(cached_catalogs["query"]["cached"])
        self.assertEqual(cached_catalogs["catalog_overlays"], original_overlays)
        validate_catalogs_response(cached_catalogs)

        # Legacy rows that still store catalog_overlays inside manifest_json also hydrate cleanly
        legacy_grid_key = "ra10.00_dec10.00_fov5.0"
        with self.cache.connection() as conn:
            conn.execute(
                """
                INSERT INTO sky_patches (
                    grid_key, ra_center, dec_center, fov_arcmin,
                    manifest_json, catalogs_json, byte_size, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    legacy_grid_key,
                    10.0,
                    10.0,
                    5.0,
                    json.dumps(manifest),
                    json.dumps(stored_catalogs),
                    1024,
                    time.time(),
                    time.time(),
                ),
            )
        legacy_manifest = self.cache.get_manifest(legacy_grid_key)
        self.assertIsNotNone(legacy_manifest)
        assert legacy_manifest is not None
        self.assertEqual(legacy_manifest["catalog_overlays"], original_overlays)

    def test_synthetic_frame_lru_cache_is_bounded_during_prewarm(self) -> None:
        synthetic_patch._FRAME_LRU_CACHE.clear()
        service = StarHuntersBackendService(cache=self.cache)
        count = service.prewarm_known_data()
        self.assertEqual(count, len(SHOWCASE_TARGETS))
        self.assertLessEqual(
            len(synthetic_patch._FRAME_LRU_CACHE),
            synthetic_patch._MAX_CACHED_FRAME_SETS,
        )

        # Behavior of get_sky_patch and get_frame_bytes remains intact for every target
        for slug, meta in SHOWCASE_TARGETS.items():
            manifest = service.get_sky_patch(meta["ra"], meta["dec"], 5.0)
            self.assertEqual(manifest["target_metadata"]["slug"], slug)
            body, headers = service.get_frame_bytes(
                manifest["query"]["grid_key"],
                "pass_1.bin",
                precision="float32",
            )
            self.assertEqual(headers["X-Binary-Dtype"], "float32")
            self.assertGreater(len(body), 0)

    def test_preset_builder_generates_validates_and_cleans_bundles(self) -> None:
        presets_dir = self.tmp_path / "presets"
        stale_file = presets_dir / "wise_0855" / "wise_fallback.bin"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_bytes(b"stale")

        generated = generate_all_synthetic_presets(output_root=presets_dir)
        self.assertEqual(set(generated), set(SHOWCASE_TARGETS))
        self.assertFalse(stale_file.exists())
        self.assertEqual(validate_written_presets(presets_dir, generated), generated)

        index_path = presets_dir / "global_known_objects.json"
        self.assertTrue(index_path.is_file())
        global_index = json.loads(index_path.read_text(encoding="utf-8"))
        object_ids = [entry["object_id"] for entry in global_index if entry.get("object_id")]
        self.assertGreater(len(object_ids), len(SHOWCASE_TARGETS))
        self.assertEqual(len(object_ids), len(set(object_ids)))

        # Validation error paths
        bad_bundle = build_synthetic_preset_bundle("wise_0855")
        bad_frames = copy.deepcopy(bad_bundle)
        bad_frames.frames["pass_1.bin"] = b"too_short"
        with self.assertRaises(ValueError):
            validate_preset_bundle(bad_frames)

        missing_frame = build_synthetic_preset_bundle("wise_0855")
        del missing_frame.frames["pass_2_flags.bin"]
        with self.assertRaises(ValueError):
            validate_preset_bundle(missing_frame)

        few_epochs = build_synthetic_preset_bundle("wise_0855")
        few_epochs.manifest["epochs"] = few_epochs.manifest["epochs"][:1]
        with self.assertRaises(ValueError):
            validate_preset_bundle(few_epochs)

        # Live bundle assembly & persistence
        live_bundle: PresetBundle = build_live_preset_bundle(
            slug="wise_0855",
            meta=SHOWCASE_TARGETS["wise_0855"],
            frames=bad_bundle.frames,
            epochs=bad_bundle.manifest["epochs"],
            summary={
                "primary_mission": "SPHEREx",
                "has_multi_epoch_spherex": True,
                "uses_wise_fallback": False,
                "baseline_days": 182.7,
                "status_note": "2 real SPHEREx comparison epoch(s)",
            },
        )
        validate_preset_bundle(live_bundle)
        write_preset_bundle(live_bundle, self.tmp_path / "live_presets")
        self.assertEqual(
            validate_written_presets(self.tmp_path / "live_presets", ["wise_0855"]),
            ["wise_0855"],
        )

    def test_preset_builder_seeding_and_summary_helpers(self) -> None:
        horizons_result = (
            "Date__(UT)__HR:MN, , ,R.A._(ICRF), DEC_(ICRF), APmag, S-brt,\n"
            "$$SOE\n"
            " 2025-May-01 00:00, , , 192.45000, +7.82000, 7.50, n.a.,\n"
            "$$EOE\n"
        )
        most_brief = (
            "|  ra_obj| dec_obj| vmag|         start_date|           end_date|"
            "  mjd_start|    mjd_end|                                 uri|\n"
            "|  double|  double|double|               char|               char|"
            "     double|     double|                                char|\n"
            "  192.45000  7.82000  7.50  2025-05-01T00:00:00  2025-05-01T00:05:00"
            "  60796.0  60796.003  test_most.fits\n"
        )

        def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "horizons.api" in url:
                return httpx.Response(200, json={"result": horizons_result})
            if "MOST" in url:
                return httpx.Response(200, text=most_brief)
            return httpx.Response(404)

        async def _run() -> tuple[
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
        ]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
                ephem = await seed_ephemeris_for_bodies(
                    self.cache,
                    [HORIZONS_FEATURED_BODIES[-2]],
                    start_date="2025-05-01",
                    stop_date="2025-05-02",
                    step_size="1d",
                    client=client,
                )
                pipeline = AutomatedCelestialPipeline(cache=self.cache, http_client=client)
                regions = await seed_showcase_regions(
                    pipeline,
                    targets={"wise_0855": SHOWCASE_TARGETS["wise_0855"]},
                    allow_live=False,
                )
                most = await seed_most_intersections(
                    self.cache,
                    ["20000001:NID"],
                    client=client,
                )
            return ephem, regions, most

        ephem_summary, regions_summary, most_summary = asyncio.run(_run())
        self.assertEqual(ephem_summary[HORIZONS_FEATURED_BODIES[-2]["name"]]["rows"], 1)
        self.assertIn("wise_0855", regions_summary)
        self.assertEqual(most_summary["20000001:NID"]["matches"], 1)

        prewarm_info = prewarm_cache_with_service(self.cache)
        self.assertEqual(prewarm_info["prewarmed_patches"], len(SHOWCASE_TARGETS))

        summary_path = self.tmp_path / "nested" / "summary.json"
        write_summary_json(summary_path, {"ephemeris": ephem_summary, **prewarm_info})
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["prewarmed_patches"], len(SHOWCASE_TARGETS))

    def test_tracking_square_colors_match_frontend_contracts(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        ts_contracts_path = repo_root / "frontend" / "src" / "types" / "contracts.ts"
        ts_text = ts_contracts_path.read_text(encoding="utf-8")
        match = re.search(
            r"export const TRACKING_SQUARE_COLORS\s*:[^=]*=\s*\[(.*?)\]\s*as const;",
            ts_text,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "Could not locate TRACKING_SQUARE_COLORS in contracts.ts")
        assert match is not None
        ts_colors = re.findall(r"'#[0-9A-Fa-f]{6}'", match.group(1))
        ts_palette = [color.strip("'") for color in ts_colors]
        self.assertEqual(ts_palette, list(TRACKING_SQUARE_COLORS))

    def test_lifespan_closes_cache_if_aclose_raises_and_forwards_client_to_most(self) -> None:
        service = StarHuntersBackendService(cache=self.cache)
        observed_clients: list[httpx.AsyncClient | None] = []

        async def _fake_most(
            *,
            obj_name: str,
            client: httpx.AsyncClient | None = None,
        ) -> dict[str, object]:
            observed_clients.append(client)
            return {"query_name": obj_name, "status": "no_match", "matches": []}

        with (
            patch(
                "app.services.backend_service.fetch_most_spherex_matches",
                side_effect=_fake_most,
            ),
            TestClient(create_app(service)) as client,
        ):
            resp = client.get("/api/v1/solar-system/most", params={"obj_name": "Ceres"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(observed_clients), 1)
            self.assertIs(observed_clients[0], service.catalog_pipeline.http_client)

        async def _run_failing_aclose() -> None:
            svc = StarHuntersBackendService(cache=self.cache)
            app = create_app(svc)
            with self.assertRaises(RuntimeError):
                async with app.router.lifespan_context(app):
                    managed = svc.catalog_pipeline.http_client
                    assert managed is not None
                    managed.aclose = AsyncMock(side_effect=RuntimeError("close boom"))  # type: ignore[method-assign]
            self.assertEqual(self.cache._connections, [])

        asyncio.run(_run_failing_aclose())

    def test_synthetic_frame_lru_cache_is_thread_safe(self) -> None:
        synthetic_patch._FRAME_LRU_CACHE.clear()
        slugs = list(SHOWCASE_TARGETS)[:6]

        def _worker(target_slug: str) -> tuple[str, int]:
            frames = generate_synthetic_epoch_frames(slug=target_slug)
            return target_slug, len(frames["pass_1.bin"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_worker, slugs * 2))

        self.assertEqual(len(results), len(slugs) * 2)
        for _slug, size in results:
            self.assertGreater(size, 0)
        self.assertLessEqual(
            len(synthetic_patch._FRAME_LRU_CACHE),
            synthetic_patch._MAX_CACHED_FRAME_SETS,
        )

    def test_scripts_generate_and_build_presets_subset_preserves_global_index(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        gen_spec = importlib.util.spec_from_file_location(
            "gen_presets_script",
            repo_root / "scripts" / "generate_synthetic_presets.py",
        )
        assert gen_spec is not None and gen_spec.loader is not None
        gen_mod = importlib.util.module_from_spec(gen_spec)
        gen_spec.loader.exec_module(gen_mod)

        fake_root = self.tmp_path / "script_repo"
        with patch.object(gen_mod, "REPO_ROOT", fake_root):
            self.assertEqual(gen_mod.main(), 0)
        self.assertTrue(
            (fake_root / "frontend" / "public" / "presets" / "global_known_objects.json").is_file()
        )

        build_spec = importlib.util.spec_from_file_location(
            "build_presets_script",
            repo_root / "scripts" / "build_presets.py",
        )
        assert build_spec is not None and build_spec.loader is not None
        build_mod = importlib.util.module_from_spec(build_spec)
        build_spec.loader.exec_module(build_mod)

        live_out = self.tmp_path / "live_subset_presets"
        sample_bundle = build_synthetic_preset_bundle("wise_0855")
        sample_frames = sample_bundle.frames
        sample_epochs = sample_bundle.manifest["epochs"]
        sample_summary = sample_bundle.manifest["data_source_summary"]

        with (
            patch.object(build_mod, "_probe_irsa", return_value=True),
            patch.object(
                build_mod,
                "_resolve_live_frames",
                return_value=(sample_frames, sample_epochs, sample_summary, sample_bundle.spectrum),
            ),
            patch.object(
                build_mod,
                "_fetch_live_catalog_overlays",
                new=AsyncMock(return_value={"known_objects": [], "gaia_dr3": []}),
            ),
        ):
            built = build_mod.build_presets(
                output_dir=live_out,
                fallback_synthetic=False,
                targets=["wise_0855"],
                live_catalogs=True,
            )

        self.assertEqual(built, ["wise_0855"])
        global_index = json.loads(
            (live_out / "global_known_objects.json").read_text(encoding="utf-8")
        )
        indexed_slugs = {
            entry.get("target_slug") for entry in global_index if entry.get("target_slug")
        }
        self.assertTrue(set(SHOWCASE_TARGETS).issubset(indexed_slugs))

    def test_sqlite_throttled_last_accessed_writes(self) -> None:
        throttled_db = str(self.tmp_path / "throttled.db")
        throttled_cache = SpatialSqliteCache(
            db_path=throttled_db,
            max_cache_bytes=64 * 1024 * 1024,
            last_access_update_interval=300.0,
        )
        try:
            meta = SHOWCASE_TARGETS["wise_0855"]
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"], dec=meta["dec"], fov_arcmin=5.0, slug="wise_0855"
            )
            frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
            grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)
            throttled_cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)

            with throttled_cache.connection() as conn:
                initial_ts = float(
                    conn.execute(
                        "SELECT last_accessed FROM sky_patches WHERE grid_key = ?",
                        (grid_key,),
                    ).fetchone()["last_accessed"]
                )

            # Reading immediately from SQLite must NOT rewrite last_accessed within 300s
            self.assertIsNotNone(throttled_cache.get_manifest(grid_key))
            throttled_cache._mem_cache.clear()
            self.assertIsNotNone(throttled_cache.get_frame(grid_key, "pass_1.bin"))
            with throttled_cache.connection() as conn:
                after_fast_read_ts = float(
                    conn.execute(
                        "SELECT last_accessed FROM sky_patches WHERE grid_key = ?",
                        (grid_key,),
                    ).fetchone()["last_accessed"]
                )
            self.assertEqual(after_fast_read_ts, initial_ts)

            # Backdate last_accessed beyond interval and clear memory cache -> read updates SQLite
            stale_ts = initial_ts - 600.0
            with throttled_cache.connection() as conn:
                conn.execute(
                    "UPDATE sky_patches SET last_accessed = ? WHERE grid_key = ?",
                    (stale_ts, grid_key),
                )
            throttled_cache._mem_cache.clear()
            self.assertIsNotNone(throttled_cache.get_manifest(grid_key))
            with throttled_cache.connection() as conn:
                refreshed_ts = float(
                    conn.execute(
                        "SELECT last_accessed FROM sky_patches WHERE grid_key = ?",
                        (grid_key,),
                    ).fetchone()["last_accessed"]
                )
            self.assertGreater(refreshed_ts, stale_ts)
        finally:
            throttled_cache.close()

    def test_in_process_lru_cache_hit_immutability_eviction_and_invalidation(self) -> None:
        tiny_mem_db = str(self.tmp_path / "mem_lru.db")
        mem_cache = SpatialSqliteCache(
            db_path=tiny_mem_db,
            max_cache_bytes=64 * 1024 * 1024,
            max_memory_bytes=2 * 1024 * 1024,
        )
        try:
            meta = SHOWCASE_TARGETS["wise_0855"]
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"], dec=meta["dec"], fov_arcmin=5.0, slug="wise_0855"
            )
            frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
            grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)
            mem_cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)

            # Immutability: mutating returned manifest must not corrupt cached copy
            m1 = mem_cache.get_manifest(grid_key)
            assert m1 is not None
            m1["query"]["grid_key"] = "mutated_by_caller"
            m2 = mem_cache.get_manifest(grid_key)
            assert m2 is not None
            self.assertEqual(m2["query"]["grid_key"], grid_key)

            # Byte-budget eviction: adding multiple 1.5 MB raw frames stays <= 2 MB cap
            self.assertIsNotNone(mem_cache.get_frame(grid_key, "pass_1.bin"))
            self.assertIsNotNone(mem_cache.get_frame(grid_key, "pass_2.bin"))
            self.assertLessEqual(mem_cache._mem_cache.current_bytes, 2 * 1024 * 1024)
            self.assertIsNone(mem_cache._mem_cache.get(f"frame_raw:{grid_key}:pass_1.bin"))

            # put_sky_patch invalidates stale frame_raw and populates manifest in memory
            self.assertIsNotNone(mem_cache.get_manifest(grid_key))
            self.assertIsNotNone(mem_cache._mem_cache.get(f"manifest:{grid_key}"))
            mem_cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)
            self.assertIsNotNone(mem_cache._mem_cache.get(f"manifest:{grid_key}"))
            self.assertIsNone(mem_cache._mem_cache.get(f"frame_raw:{grid_key}:pass_2.bin"))

            # clear() wipes memory cache too
            self.assertIsNotNone(mem_cache.get_manifest(grid_key))
            self.assertGreater(mem_cache._mem_cache.entry_count, 0)
            mem_cache.clear()
            self.assertEqual(mem_cache._mem_cache.entry_count, 0)

            # SQLite _prune_if_needed invalidates memory entries for evicted patches & regions
            mem_cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)
            self.assertIsNotNone(mem_cache.get_manifest(grid_key))
            mem_cache.put_cached_catalog_region("simbad", 10.0, 10.0, 0.1, {"rows": [1, 2, 3]})
            self.assertIsNotNone(mem_cache.get_cached_catalog_region("simbad", 10.0, 10.0, 0.1))

            one_size = mem_cache.get_patch_byte_size(grid_key)
            mem_cache.max_cache_bytes = int(one_size * 1.2)
            other_manifest = build_sky_patch_manifest_payload(
                ra=55.0, dec=22.0, fov_arcmin=5.0, slug="custom_sky"
            )
            other_frames = generate_synthetic_epoch_frames(
                slug="custom_sky", manifest=other_manifest
            )
            other_key, o_ra, o_dec = compute_grid_key(55.0, 22.0, 5.0)
            mem_cache.put_sky_patch(other_key, o_ra, o_dec, 5.0, other_manifest, other_frames)

            self.assertIsNone(mem_cache._mem_cache.get(f"manifest:{grid_key}"))
            self.assertIsNone(mem_cache.get_manifest(grid_key))
        finally:
            mem_cache.close()

    def test_pre_gzipped_float16_frame_endpoint_and_identity_fallback(self) -> None:
        service = StarHuntersBackendService(cache=self.cache)
        manifest = service.get_sky_patch(133.7951, -7.2451, 5.0)
        grid_key = manifest["query"]["grid_key"]
        client = TestClient(create_app(service))

        # Float16 is converted on-the-fly from cached float32 without polluting RAM LRU with float16
        with patch(
            "app.services.backend_service.convert_float32_bytes_to_float16_bytes",
            wraps=convert_float32_bytes_to_float16_bytes,
        ) as mock_f16:
            r1 = client.get(
                f"/api/v1/frames/{grid_key}/pass_1.bin",
                params={"precision": "float16"},
                headers={"Accept-Encoding": "gzip"},
            )
            r2 = client.get(
                f"/api/v1/frames/{grid_key}/pass_1.bin",
                params={"precision": "float16"},
                headers={"Accept-Encoding": "gzip"},
            )
            self.cache._mem_cache.clear()
            r3 = client.get(
                f"/api/v1/frames/{grid_key}/pass_1.bin",
                params={"precision": "float16"},
                headers={"Accept-Encoding": "gzip"},
            )
            self.assertEqual(mock_f16.call_count, 3)

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r1.headers.get("content-encoding"), "gzip")
        self.assertEqual(r2.headers.get("content-encoding"), "gzip")
        self.assertEqual(r3.headers.get("content-encoding"), "gzip")
        self.assertIn("Accept-Encoding", r1.headers.get("vary", ""))
        self.assertEqual(r1.headers.get("x-binary-dtype"), "float16")
        self.assertEqual(len(r1.content), BINARY_DETECTOR_FLOAT16_BYTES)
        self.assertEqual(r1.content, r2.content)
        self.assertEqual(r1.content, r3.content)
        # _mem_cache holds canonical float32 only—never float16 or uncompressed frame_raw
        self.assertIsNotNone(self.cache._mem_cache.get(f"frame_gz:{grid_key}:pass_1.bin:float32"))
        self.assertIsNone(self.cache._mem_cache.get(f"frame_gz:{grid_key}:pass_1.bin:float16"))
        self.assertIsNone(self.cache._mem_cache.get(f"frame_raw:{grid_key}:pass_1.bin"))

        identity_resp = client.get(
            f"/api/v1/frames/{grid_key}/pass_1.bin",
            params={"precision": "float16"},
            headers={"Accept-Encoding": "identity"},
        )
        self.assertEqual(identity_resp.status_code, 200)
        self.assertIsNone(identity_resp.headers.get("content-encoding"))
        self.assertEqual(len(identity_resp.content), BINARY_DETECTOR_FLOAT16_BYTES)
        self.assertEqual(identity_resp.content, r1.content)

    def test_cached_static_files_headers(self) -> None:
        dist_dir = self.tmp_path / "dist"
        assets_dir = dist_dir / "assets"
        assets_dir.mkdir(parents=True)
        (dist_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
        (assets_dir / "app-abc123.js").write_text("console.log('ok');", encoding="utf-8")

        static_app = FastAPI()
        static_app.mount("/", CachedStaticFiles(directory=str(dist_dir), html=True), name="spa")
        client = TestClient(static_app)

        asset_resp = client.get("/assets/app-abc123.js")
        self.assertEqual(asset_resp.status_code, 200)
        self.assertEqual(
            asset_resp.headers.get("cache-control"),
            "public, max-age=31536000, immutable",
        )

        root_resp = client.get("/")
        self.assertEqual(root_resp.status_code, 200)
        self.assertEqual(root_resp.headers.get("cache-control"), "no-cache")

        html_resp = client.get("/index.html")
        self.assertEqual(html_resp.status_code, 200)
        self.assertEqual(html_resp.headers.get("cache-control"), "no-cache")

    def test_southern_sky_synthetic_cutout_and_async_earth_to_sky_background_prewarm(
        self,
    ) -> None:
        from app.services.spherex_synthetic import generate_synthetic_spherex_cutout

        # Southern-sky coordinates where ra + dec < 0 must not raise ValueError on seed=None
        cutout = generate_synthetic_spherex_cutout(
            ra=10.25,
            dec=-65.5,
            fov_arcmin=5.0,
            bandpass="SPHEREx-D1",
            shape=(32, 32),
            seed=None,
        )
        self.assertEqual(cutout.clean_image.shape, (32, 32))

        async def _verify_no_speculative_prewarm_by_default() -> None:
            service = StarHuntersBackendService(cache=self.cache, enable_archival_data=False)
            payload = await service.get_earth_to_sky_async(
                lat=None,
                lon=None,
                iso_time="2026-11-21T12:00:00Z",
                fallback_seed=11,
            )
            self.assertTrue(payload["earth"]["is_fallback_capital"])
            if service._background_tasks:
                await asyncio.gather(*list(service._background_tasks))
            gk, _, _ = compute_grid_key(
                float(payload["zenith_icrs"]["ra"]),
                float(payload["zenith_icrs"]["dec"]),
                5.0,
            )
            self.assertIsNone(self.cache.get_manifest(gk))

        asyncio.run(_verify_no_speculative_prewarm_by_default())

        from app.services.backend_service import compute_max_concurrent_coadds

        # Readable ~2/3 formula: reserved_for_io = max(1, total // 3)
        self.assertEqual(compute_max_concurrent_coadds(1), 1)
        self.assertEqual(compute_max_concurrent_coadds(2), 1)
        self.assertEqual(compute_max_concurrent_coadds(4), 3)
        self.assertEqual(compute_max_concurrent_coadds(6), 4)
        self.assertEqual(compute_max_concurrent_coadds(7), 5)
        self.assertEqual(compute_max_concurrent_coadds(8), 6)
        self.assertEqual(compute_max_concurrent_coadds(10), 7)
        self.assertEqual(compute_max_concurrent_coadds(12), 8)
        self.assertEqual(compute_max_concurrent_coadds(16), 11)

    def test_dual_precision_sqlite_encoding_and_zero_recompression(self) -> None:
        meta = SHOWCASE_TARGETS["wise_0855"]
        manifest = build_sky_patch_manifest_payload(
            ra=meta["ra"], dec=meta["dec"], fov_arcmin=5.0, slug="wise_0855"
        )
        frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
        grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)
        self.cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)

        with self.cache.connection() as conn:
            rows = conn.execute(
                "SELECT filename, encoding FROM binary_frames WHERE grid_key = ?",
                (grid_key,),
            ).fetchall()
        encodings_by_name: dict[str, set[str]] = {}
        for r in rows:
            encodings_by_name.setdefault(str(r["filename"]), set()).add(str(r["encoding"]))
        self.assertEqual(encodings_by_name["pass_1.bin"], {"gzip_float32"})
        self.assertEqual(encodings_by_name["pass_1_flags.bin"], {"gzip_uint8"})

        raw32 = self.cache.get_frame(grid_key, "pass_1.bin")
        assert raw32 is not None
        self.assertEqual(raw32, frames["pass_1.bin"])
        flags_raw = self.cache.get_frame(grid_key, "pass_1_flags.bin")
        assert flags_raw is not None
        self.assertEqual(flags_raw, frames["pass_1_flags.bin"])

        self.cache._mem_cache.clear()
        with patch("app.services.spatial_cache.gzip.compress") as mock_gz:
            gz32 = self.cache.get_frame_gzip(grid_key, "pass_1.bin", precision="float32")
            gz16 = self.cache.get_frame_gzip(grid_key, "pass_1.bin", precision="float16")
            self.assertEqual(mock_gz.call_count, 0)

        assert gz32 is not None
        self.assertIsNone(gz16)
        self.assertEqual(gz32[1], "float32")
        self.assertEqual(gzip.decompress(gz32[0]), frames["pass_1.bin"])

    def test_sky_patch_and_spectrum_7day_ttl_and_stale_if_error(self) -> None:
        from app.services.spatial_cache import SpatialSqliteCache

        ttl_cache = SpatialSqliteCache(
            db_path=str(self.tmp_path / "ttl_cache.db"),
            patch_ttl_seconds=60.0,
        )
        try:
            meta = SHOWCASE_TARGETS["wise_0855"]
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"], dec=meta["dec"], fov_arcmin=5.0, slug="wise_0855"
            )
            manifest["data_source_summary"]["provenance_tier"] = "primary"
            manifest["data_source_summary"]["status_note"] = "SPHEREx QR2 archival passes"
            frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
            grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)
            ttl_cache.put_sky_patch(grid_key, ra_q, dec_q, 5.0, manifest, frames)

            self.assertIsNotNone(ttl_cache.get_manifest(grid_key))
            # Age created_at past the 60s TTL
            with ttl_cache.connection() as conn:
                conn.execute(
                    "UPDATE sky_patches SET created_at = ? WHERE grid_key = ?",
                    (time.time() - 120.0, grid_key),
                )
            ttl_cache._mem_cache.clear()
            self.assertIsNone(ttl_cache.get_manifest(grid_key, allow_stale=False))
            stale = ttl_cache.get_manifest(grid_key, allow_stale=True)
            assert stale is not None
            self.assertEqual(stale["data_source_summary"]["provenance_tier"], "primary")

            # Stale-if-error in StarHuntersBackendService when upstream degrades to simulated
            svc = StarHuntersBackendService(cache=ttl_cache, enable_archival_data=True)
            with patch.object(
                svc.archival_resolver,
                "resolve_archival_patch_sync",
                side_effect=RuntimeError("IRSA unreachable"),
            ):
                served = svc.get_sky_patch(meta["ra"], meta["dec"], 5.0)
                self.assertEqual(served["data_source_summary"]["provenance_tier"], "primary")
        finally:
            ttl_cache.close()

    def test_cpu_and_io_executor_isolation(self) -> None:
        async def _verify_isolation() -> None:
            seed_svc = StarHuntersBackendService(cache=self.cache, enable_archival_data=False)
            warm = await seed_svc.get_sky_patch_async(133.7951, -7.2451, 5.0, live_catalogs=False)
            warm_gk = warm["query"]["grid_key"]
            self.cache._mem_cache.clear()

            from app.services.skyview_fallback import SkyViewFallbackError
            from app.services.wise_fallback import WiseFallbackError

            def _raise_wise(**_kw: object) -> object:
                raise WiseFallbackError("offline")

            def _raise_skyview(**_kw: object) -> object:
                raise SkyViewFallbackError("offline")

            with patch.dict(os.environ, {"ARCHIVAL_MAX_CONCURRENT_COADDS": "1"}):
                svc = StarHuntersBackendService(
                    cache=self.cache,
                    enable_archival_data=True,
                    spherex_cutout_fetcher=lambda **_kw: (time.sleep(0.35), [])[1],
                    wise_fallback_fetcher=_raise_wise,
                    skyview_fallback_fetcher=_raise_skyview,
                    cold_miss_rate_limit_per_min=0,
                )

            slow_tasks = [
                asyncio.create_task(
                    svc.get_sky_patch_async(40.0 + i * 2.0, 10.0, 5.0, live_catalogs=False)
                )
                for i in range(4)
            ]
            await asyncio.sleep(0.03)
            t0 = time.perf_counter()
            e2s, (frame_body, _hdr) = await asyncio.gather(
                svc.get_earth_to_sky_async(lat=51.5, lon=-0.12, prewarm_patch=False),
                svc.get_frame_bytes_async(
                    warm_gk, "pass_1.bin", precision="float16", accept_gzip=True
                ),
            )
            io_elapsed_s = time.perf_counter() - t0
            self.assertIn("zenith_icrs", e2s)
            self.assertGreater(len(frame_body), 0)
            self.assertLess(io_elapsed_s, 0.25)
            await asyncio.gather(*slow_tasks)

        asyncio.run(_verify_isolation())

    def test_cold_miss_per_ip_rate_limiter_allows_cached_reads(self) -> None:
        service = StarHuntersBackendService(
            cache=self.cache,
            enable_archival_data=False,
            cold_miss_rate_limit_per_min=2,
        )
        # Pre-seed one cached patch before creating client
        cached_manifest = service.get_sky_patch(133.7951, -7.2451, 5.0)
        self.assertIsNotNone(cached_manifest)
        client = TestClient(create_app(service))

        bot_headers = {"fly-client-ip": "203.0.113.10"}
        human_headers = {"fly-client-ip": "203.0.113.99"}

        r1 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 20.0, "dec": 10.0, "live_catalogs": "false"},
            headers=bot_headers,
        )
        r2 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 25.0, "dec": 12.0, "live_catalogs": "false"},
            headers=bot_headers,
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

        # 3rd cold miss from same IP hits 429 with Retry-After
        r3_cold = client.get(
            "/api/v1/sky-patch",
            params={"ra": 30.0, "dec": 14.0, "live_catalogs": "false"},
            headers=bot_headers,
        )
        self.assertEqual(r3_cold.status_code, 429)
        self.assertIn("retry-after", {k.lower() for k in r3_cold.headers})

        # Cached read from that same IP is never throttled
        r4_cached = client.get(
            "/api/v1/sky-patch",
            params={"ra": 133.7951, "dec": -7.2451, "live_catalogs": "false"},
            headers=bot_headers,
        )
        self.assertEqual(r4_cached.status_code, 200)

        # Different IP still has its own fresh cold-miss quota
        r5_other_ip = client.get(
            "/api/v1/sky-patch",
            params={"ra": 30.0, "dec": 14.0, "live_catalogs": "false"},
            headers=human_headers,
        )
        self.assertEqual(r5_other_ip.status_code, 200)

    def test_extract_client_ip_supports_agnostic_proxy_headers(self) -> None:
        from starlette.requests import Request

        from app.main import _extract_client_ip

        def make_request(
            headers: dict[str, str] | None = None,
            client: tuple[str, int] | None = ("127.0.0.1", 12345),
        ) -> Request:
            raw_headers = [
                (k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in (headers or {}).items()
            ]
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": raw_headers,
                "client": client,
            }
            return Request(scope)

        self.assertEqual(
            _extract_client_ip(make_request({"CF-Connecting-IP": "198.51.100.7"})),
            "198.51.100.7",
        )
        self.assertEqual(
            _extract_client_ip(make_request({"X-Real-IP": "198.51.100.8:443"})),
            "198.51.100.8",
        )
        self.assertEqual(
            _extract_client_ip(
                make_request({"Forwarded": 'for="[2001:db8:cafe::17]:4711";proto=https'})
            ),
            "2001:db8:cafe::17",
        )
        self.assertEqual(
            _extract_client_ip(
                make_request({"X-Forwarded-For": "not-an-ip, 203.0.113.42, 10.0.0.1"})
            ),
            "203.0.113.42",
        )
        self.assertEqual(
            _extract_client_ip(make_request(client=("192.0.2.5", 5000))),
            "192.0.2.5",
        )
        self.assertEqual(
            _extract_client_ip(make_request(client=None)),
            "unknown",
        )

    def test_sqlite_schema_user_version_and_last_accessed_index(self) -> None:
        from app.services.spatial_cache import SCHEMA_VERSION

        with self.cache.connection() as conn:
            ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(ver, SCHEMA_VERSION)
            cols = {
                str(r["name"]) for r in conn.execute("PRAGMA table_info(binary_frames)").fetchall()
            }
            patch_idx = {
                str(r["name"]) for r in conn.execute("PRAGMA index_list(sky_patches)").fetchall()
            }
            cat_idx = {
                str(r["name"])
                for r in conn.execute("PRAGMA index_list(catalog_region_cache)").fetchall()
            }
        self.assertIn("encoding", cols)
        self.assertIn("idx_sky_patches_eviction", patch_idx)
        self.assertIn("idx_catalog_region_last_accessed", cat_idx)

    def test_memory_lru_o1_tag_invalidation(self) -> None:
        gk1 = "ra10.00_dec20.00_fov5.0"
        gk2 = "ra50.00_dec-10.00_fov5.0"
        self.cache._mem_cache.put(f"manifest:{gk1}", b"{}")
        self.cache._mem_cache.put(f"frame_gz:{gk1}:pass_1.bin:float16", b"abc")
        self.cache._mem_cache.put(f"manifest:{gk2}", b"{}")

        self.cache._invalidate_grid_key_memory(gk1)
        self.assertIsNone(self.cache._mem_cache.get(f"manifest:{gk1}"))
        self.assertIsNone(self.cache._mem_cache.get(f"frame_gz:{gk1}:pass_1.bin:float16"))
        self.assertNotIn(gk1, self.cache._mem_cache._tag_to_keys)
        self.assertIsNotNone(self.cache._mem_cache.get(f"manifest:{gk2}"))

    def test_aging_priority_cpu_gate_spectrum_fast_lane_and_starvation_mode(self) -> None:
        from app.services.backend_service import (
            CPU_PRIORITY_SKY_PATCH,
            CPU_PRIORITY_SPECTRUM,
            _AgingPrioritySemaphore,
        )

        async def _verify_gate() -> None:
            gate = _AgingPrioritySemaphore(capacity=1, starvation_threshold_s=0.06)
            order: list[str] = []

            async def _task(tag: str, prio: int, hold: float = 0.03) -> None:
                async with gate.slot(prio):
                    order.append(tag)
                    await asyncio.sleep(hold)

            holder = asyncio.create_task(_task("holder", CPU_PRIORITY_SKY_PATCH, hold=0.04))
            await asyncio.sleep(0.005)
            p1 = asyncio.create_task(_task("patch_1", CPU_PRIORITY_SKY_PATCH, hold=0.04))
            s1 = asyncio.create_task(_task("spec_1", CPU_PRIORITY_SPECTRUM, hold=0.04))
            await asyncio.sleep(0.07)  # patch_1 ages past 0.06s while holder + spec_1 run
            s2 = asyncio.create_task(_task("spec_2", CPU_PRIORITY_SPECTRUM, hold=0.01))
            await asyncio.gather(holder, p1, s1, s2)
            # spec_1 jumps ahead of unstarved patch_1; then aged patch_1 beats newly arrived spec_2
            self.assertEqual(order, ["holder", "spec_1", "patch_1", "spec_2"])

        asyncio.run(_verify_gate())

    def test_search_known_objects_lru_cache_and_immutability(self) -> None:
        from app.services.object_search import _search_known_objects_cached, search_known_objects

        _search_known_objects_cached.cache_clear()
        res1 = search_known_objects("Saturn", limit=10)
        res2 = search_known_objects("  saturn ", limit=10)
        self.assertGreater(len(res1), 0)
        self.assertEqual(res1, res2)
        self.assertGreaterEqual(_search_known_objects_cached.cache_info().hits, 1)

        # Mutating caller copy must not corrupt cached result
        orig_label = res1[0]["label"]
        res1[0]["label"] = "MUTATED"
        res3 = search_known_objects("saturn", limit=10)
        self.assertEqual(res3[0]["label"], orig_label)

    def test_migration_001_to_002_drops_float16_and_backfills_popularity(self) -> None:
        import sqlite3
        from pathlib import Path

        from app.services.spatial_cache import SCHEMA_VERSION, SpatialSqliteCache

        mig_db = self.tmp_path / "v1_upgrade.db"
        sql_001 = (
            Path(__file__).resolve().parent.parent
            / "app"
            / "services"
            / "migrations"
            / "001_initial_schema.sql"
        ).read_text(encoding="utf-8")
        conn = sqlite3.connect(str(mig_db))
        conn.executescript(sql_001)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            """
            INSERT INTO sky_patches (
                grid_key, ra_center, dec_center, fov_arcmin,
                manifest_json, catalogs_json, byte_size, created_at, last_accessed
            ) VALUES (
                'ra10.00_dec20.00_fov5.0', 10.0, 20.0, 5.0,
                '{}', '{}', 500, 1750000000.0, 1750000000.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO binary_frames (grid_key, filename, encoding, compressed_blob, raw_byte_size)
            VALUES ('ra10.00_dec20.00_fov5.0', 'pass_1.bin', 'gzip_float32', X'010203', 100),
                   ('ra10.00_dec20.00_fov5.0', 'pass_1.bin', 'gzip_float16', X'0405', 50)
            """
        )
        conn.commit()
        conn.close()

        upgraded = SpatialSqliteCache(db_path=str(mig_db))
        try:
            with upgraded.connection() as c2:
                ver = int(c2.execute("PRAGMA user_version").fetchone()[0])
                self.assertEqual(ver, SCHEMA_VERSION)
                encs = [
                    str(r["encoding"])
                    for r in c2.execute("SELECT encoding FROM binary_frames").fetchall()
                ]
                self.assertEqual(encs, ["gzip_float32"])
                row = c2.execute(
                    "SELECT byte_size, distinct_visitors, popularity_score, last_popularity_ts "
                    "FROM sky_patches WHERE grid_key = 'ra10.00_dec20.00_fov5.0'"
                ).fetchone()
                self.assertEqual(int(row["byte_size"]), 7)  # "{}" (2) + "{}" (2) + X'010203' (3)
                self.assertEqual(int(row["distinct_visitors"]), 1)
                self.assertEqual(float(row["popularity_score"]), 1.0)
                self.assertEqual(float(row["last_popularity_ts"]), 1750000000.0)
        finally:
            upgraded.close()

    def test_slru_visitor_dedup_and_5_month_decay_eviction(self) -> None:
        import math

        from app.services.spatial_cache import POPULARITY_EPOCH_T0, SpatialSqliteCache

        slru_cache = SpatialSqliteCache(
            db_path=str(self.tmp_path / "decay_slru.db"),
            max_cache_bytes=50_000_000,
        )
        try:
            meta = SHOWCASE_TARGETS["wise_0855"]
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"], dec=meta["dec"], fov_arcmin=5.0, slug="wise_0855"
            )
            frames = generate_synthetic_epoch_frames(slug="wise_0855", manifest=manifest)
            gk_old_hot, ra1, dec1 = compute_grid_key(10.0, 10.0, 5.0)
            gk_new_hot, ra2, dec2 = compute_grid_key(20.0, 20.0, 5.0)

            tok_a = slru_cache.compute_visitor_token("10.0.0.1", "BrowserA")
            tok_b = slru_cache.compute_visitor_token("10.0.0.2", "BrowserB")
            assert tok_a is not None and tok_b is not None

            slru_cache.put_sky_patch(
                gk_old_hot, ra1, dec1, 5.0, manifest, frames, visitor_token=tok_a
            )
            # Same visitor hitting 10 times does NOT increment distinct_visitors
            for _ in range(10):
                self.assertFalse(slru_cache.record_patch_hit(gk_old_hot, tok_a))
            # Different visitor promotes distinct_visitors to 2
            self.assertTrue(slru_cache.record_patch_hit(gk_old_hot, tok_b))

            # Simulate gk_old_hot having 50 hits 150 days (~5 months) ago
            now = time.time()
            old_ts = now - (150 * 86400.0)
            old_rank = math.log(50.0) + slru_cache._decay_lambda * (old_ts - POPULARITY_EPOCH_T0)
            with slru_cache.connection() as conn:
                conn.execute(
                    """
                    UPDATE sky_patches
                    SET distinct_visitors = 50,
                        popularity_score = 50.0,
                        last_popularity_ts = ?,
                        popularity_rank = ?,
                        last_accessed = ?
                    WHERE grid_key = ?
                    """,
                    (old_ts, old_rank, old_ts, gk_old_hot),
                )

            # Insert gk_new_hot with 2 visitors today
            slru_cache.put_sky_patch(
                gk_new_hot, ra2, dec2, 5.0, manifest, frames, visitor_token=tok_a
            )
            self.assertTrue(slru_cache.record_patch_hit(gk_new_hot, tok_b))

            # Cap cache to hold ~2.15 patches and insert a 3rd patch -> must evict gk_old_hot
            size_new = slru_cache.get_patch_byte_size(gk_new_hot)
            slru_cache.max_cache_bytes = int(size_new * 2.15)
            gk_third, ra3, dec3 = compute_grid_key(30.0, 30.0, 5.0)
            slru_cache.put_sky_patch(
                gk_third, ra3, dec3, 5.0, manifest, frames, visitor_token=tok_a
            )

            self.assertIsNone(slru_cache.get_manifest(gk_old_hot))
            self.assertIsNotNone(slru_cache.get_manifest(gk_new_hot))
            self.assertIsNotNone(slru_cache.get_manifest(gk_third))
        finally:
            slru_cache.close()

    def test_cache_stats_endpoint_telemetry_and_token_gate(self) -> None:
        service = StarHuntersBackendService(cache=self.cache, enable_archival_data=False)
        client = TestClient(create_app(service))

        # Cold miss + piggybacked client delta
        r1 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 12.0, "dec": 14.0, "fov": 5.0, "live_catalogs": False},
            headers={
                "X-Forwarded-For": "10.1.1.1",
                "User-Agent": "UA-1",
                "X-Cache-Delta": "p=3,c=2",
            },
        )
        self.assertEqual(r1.status_code, 200)
        # Same browser repeat
        r2 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 12.0, "dec": 14.0, "fov": 5.0, "live_catalogs": False},
            headers={"X-Forwarded-For": "10.1.1.1", "User-Agent": "UA-1"},
        )
        self.assertEqual(r2.status_code, 200)
        # Second browser -> common_multi_user_hits
        r3 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 12.0, "dec": 14.0, "fov": 5.0, "live_catalogs": False},
            headers={"X-Forwarded-For": "10.1.1.2", "User-Agent": "UA-2"},
        )
        self.assertEqual(r3.status_code, 200)
        # Newer single-visitor patch must sort BELOW the 2-visitor patch in top_patches
        r4 = client.get(
            "/api/v1/sky-patch",
            params={"ra": 40.0, "dec": 40.0, "fov": 5.0, "live_catalogs": False},
            headers={"X-Forwarded-For": "10.1.1.3", "User-Agent": "UA-3"},
        )
        self.assertEqual(r4.status_code, 200)

        stats_resp = client.get("/api/v1/cache-stats")
        self.assertEqual(stats_resp.status_code, 200)
        body = stats_resp.json()
        tel = body["telemetry"]
        self.assertEqual(tel["client_local_patch_hits"], 3)
        self.assertEqual(tel["client_local_catalog_hits"], 2)
        self.assertEqual(tel["unique_cold_misses"], 2)
        self.assertEqual(tel["same_user_backend_repeats"], 1)
        self.assertEqual(tel["common_multi_user_hits"], 1)
        self.assertEqual(body["top_patches"][0]["distinct_visitors"], 2)

        with patch.dict("os.environ", {"CACHE_STATS_TOKEN": "secret-bench-token"}):
            self.assertEqual(client.get("/api/v1/cache-stats").status_code, 403)
            ok = client.get(
                "/api/v1/cache-stats",
                headers={"Authorization": "Bearer secret-bench-token"},
            )
            self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
