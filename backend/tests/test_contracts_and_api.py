"""Unit tests for Phase 0 shared contracts, 3G binary buffers, SQLite cache & backend routes."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import numpy as np

from app.main import app, create_app
from app.models.contracts import (
    BINARY_DETECTOR_FLOAT16_BYTES,
    BINARY_DETECTOR_FLOAT32_BYTES,
    BINARY_FLAGS_UINT8_BYTES,
    SHOWCASE_TARGETS,
    TRACKING_SQUARE_COLORS,
    WORLD_CAPITAL_CITIES,
    convert_float16_bytes_to_float32_bytes,
    convert_float32_bytes_to_float16_bytes,
    validate_catalogs_response,
    validate_detector_binary_buffer,
    validate_earth_to_sky_response,
    validate_flags_binary_buffer,
    validate_sky_patch_manifest,
    validate_spectrum_payload,
)
from app.services.backend_service import StarHuntersBackendService
from app.services.object_epochs import cluster_most_matches_into_passes
from app.services.spatial_cache import SpatialSqliteCache, compute_grid_key
from app.services.spectrum_extractor import SpectrumExtractionError, extract_forced_spectrum
from app.services.synthetic_patch import (
    build_sky_patch_manifest_payload,
    build_spectrum_payload,
    generate_synthetic_epoch_frames,
)
from app.services.zenith_bridge import compute_zenith_icrs


class TestPhase0ContractsAndBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_cache.db")
        self.cache = SpatialSqliteCache(db_path=self.db_path, max_cache_bytes=50 * 1024 * 1024)
        self.service = StarHuntersBackendService(cache=self.cache)

    def _make_spectral_cutout(
        self,
        ra: float,
        dec: float,
        *,
        bandpass: str = "SPHEREx-D1",
        wavelength: float | None = None,
        source_flux: float = 5.0,
    ) -> SimpleNamespace:
        header = {
            "NAXIS": 2,
            "NAXIS1": 31,
            "NAXIS2": 31,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": ra,
            "CRVAL2": dec,
            "CRPIX1": 16.0,
            "CRPIX2": 16.0,
            "CDELT1": -0.0001,
            "CDELT2": 0.0001,
        }
        if wavelength is not None:
            header["WAVELEN"] = float(wavelength)
        image = np.full((31, 31), 10.0, dtype=np.float32)
        image[13:18, 13:18] += source_flux
        variance = np.ones((31, 31), dtype=np.float32)
        flags = np.zeros((31, 31), dtype=np.uint8)
        return SimpleNamespace(
            clean_image=image,
            variance=variance,
            weight_map=np.ones_like(image),
            flags_mask=flags,
            mjd_obs=60000.0,
            bandpass=bandpass,
            header=header,
        )

    def tearDown(self) -> None:
        self.cache.close()
        self.temp_dir.cleanup()

    def test_health_endpoint_contract(self) -> None:
        res = self.service.health()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["service"], "star-hunters-backend")

    def test_quantized_grid_key_sharing(self) -> None:
        # Two queries within 0.02 deg should map to the same 0.05-deg quantized grid key
        key1, _, _ = compute_grid_key(133.7951, -7.2451, 5.0)
        key2, _, _ = compute_grid_key(133.8090, -7.2580, 5.0)
        self.assertEqual(key1, key2)
        self.assertEqual(key1, "ra133.80_dec-7.25_fov5.0")

    def test_all_showcase_manifests_and_known_objects_contract(self) -> None:
        for slug, meta in SHOWCASE_TARGETS.items():
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"],
                dec=meta["dec"],
                fov_arcmin=5.0,
                slug=slug,
            )
            parsed = validate_sky_patch_manifest(manifest)
            self.assertEqual(parsed.target_metadata.slug, slug)
            self.assertEqual(parsed.wcs.width, 256)
            self.assertEqual(parsed.wcs.height, 256)

            known = parsed.catalog_overlays.known_objects
            self.assertGreaterEqual(len(known), 2)
            # At least 1 notable mover should be default_checked=True, and background stars False
            checked = [obj for obj in known if obj.default_checked]
            unchecked = [obj for obj in known if not obj.default_checked]
            self.assertGreaterEqual(len(checked), 1)
            self.assertGreaterEqual(len(unchecked), 1)
            for obj in known:
                self.assertEqual(len(obj.epoch_positions), len(parsed.epochs))

    def test_invalid_manifest_rejected(self) -> None:
        manifest = build_sky_patch_manifest_payload(133.7951, -7.2451, 5.0, slug="wise_0855")
        bad = copy.deepcopy(manifest)
        bad["wcs"]["width"] = 128  # Must be 256
        with self.assertRaises(jsonschema.ValidationError):
            validate_sky_patch_manifest(bad)

    def test_catalogs_contract_endpoint(self) -> None:
        catalogs = self.service.get_catalogs(ra=133.7951, dec=-7.2451, fov=5.0)
        validated = validate_catalogs_response(catalogs)
        self.assertEqual(validated.wcs.width, 256)
        self.assertGreater(len(validated.catalog_overlays.known_objects), 0)

    def test_binary_frame_sizes_and_float16_3g_conversion(self) -> None:
        frames = generate_synthetic_epoch_frames("wise_0855")
        p1 = frames["pass_1.bin"]
        f1 = frames["pass_1_flags.bin"]

        count, min_v, max_v = validate_detector_binary_buffer(p1, precision="float32")
        self.assertEqual(len(p1), BINARY_DETECTOR_FLOAT32_BYTES)
        self.assertEqual(count, 6 * 256 * 256)
        self.assertGreater(max_v, min_v)

        f16 = convert_float32_bytes_to_float16_bytes(p1)
        self.assertEqual(len(f16), BINARY_DETECTOR_FLOAT16_BYTES)
        validate_detector_binary_buffer(f16, precision="float16")

        roundtrip_f32 = convert_float16_bytes_to_float32_bytes(f16)
        self.assertEqual(len(roundtrip_f32), BINARY_DETECTOR_FLOAT32_BYTES)

        f_len, nonzero = validate_flags_binary_buffer(f1)
        self.assertEqual(f_len, BINARY_FLAGS_UINT8_BYTES)
        self.assertGreater(nonzero, 0)

        # Verify float16 clamping bounds and NaN/Inf/length validation errors
        extreme = np.zeros(6 * 256 * 256, dtype="<f4")
        extreme[0] = 1e5
        extreme[1] = -1e5
        clamped_f16 = np.frombuffer(
            convert_float32_bytes_to_float16_bytes(extreme.tobytes()),
            dtype="<f2",
        )
        self.assertEqual(float(clamped_f16[0]), 65504.0)
        self.assertEqual(float(clamped_f16[1]), -65504.0)

        nan_buf = np.zeros(6 * 256 * 256, dtype="<f4")
        nan_buf[10] = np.nan
        with self.assertRaises(ValueError):
            validate_detector_binary_buffer(nan_buf.tobytes(), precision="float32")
        with self.assertRaises(ValueError):
            validate_detector_binary_buffer(b"short", precision="float32")
        with self.assertRaises(ValueError):
            validate_flags_binary_buffer(b"short")

    def test_sqlite_spatial_cache_hit_and_lru_pruning(self) -> None:
        import time
        import warnings

        # Verify WAL mode is active on SQLite connections
        with self.cache.connection() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            fk_enabled = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(str(journal_mode).lower(), "wal")
            self.assertEqual(int(fk_enabled), 1)

        # First call: cache miss (cached=False)
        m1 = self.service.get_sky_patch(133.7951, -7.2451, 5.0)
        self.assertFalse(m1["query"]["cached"])

        # Second call within 0.02 deg: cache hit (cached=True) and SQLite lookup in <20 ms
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            t0 = time.perf_counter()
            direct_cached = self.cache.get_manifest(m1["query"]["grid_key"])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            m2 = self.service.get_sky_patch(133.8010, -7.2490, 5.0)
        self.assertIsNotNone(direct_cached)
        self.assertTrue(direct_cached["query"]["cached"])
        self.assertTrue(m2["query"]["cached"])
        self.assertEqual(m1["query"]["grid_key"], m2["query"]["grid_key"])
        self.assertLess(elapsed_ms, 20.0)
        self.assertEqual([w for w in caught if issubclass(w.category, ResourceWarning)], [])

        # Frame streaming in float32 and float16 with immutable Cache-Control
        grid_key = m1["query"]["grid_key"]
        body32, headers32 = self.service.get_frame_bytes(
            grid_key, "pass_1.bin", precision="float32"
        )
        self.assertEqual(len(body32), BINARY_DETECTOR_FLOAT32_BYTES)
        self.assertIn("immutable", headers32["Cache-Control"])
        self.assertEqual(headers32["X-Binary-Dtype"], "float32")

        body16, headers16 = self.service.get_frame_bytes(
            grid_key, "pass_1.bin", precision="float16"
        )
        self.assertEqual(len(body16), BINARY_DETECTOR_FLOAT16_BYTES)
        self.assertEqual(headers16["X-Binary-Dtype"], "float16")

    def test_sqlite_lru_eviction_under_small_cap(self) -> None:
        # Measure compressed size of one patch, then cap cache at 1.4x one patch
        tiny_db = str(Path(self.temp_dir.name) / "tiny.db")
        tiny_cache = SpatialSqliteCache(db_path=tiny_db, max_cache_bytes=10_000_000)
        try:
            svc = StarHuntersBackendService(cache=tiny_cache)

            m_a = svc.get_sky_patch(10.0, 10.0, 5.0)
            grid_a = m_a["query"]["grid_key"]
            one_patch_bytes = tiny_cache.get_patch_byte_size(grid_a)
            self.assertGreater(one_patch_bytes, 0)
            tiny_cache.max_cache_bytes = int(one_patch_bytes * 1.4)

            m_b = svc.get_sky_patch(50.0, 20.0, 5.0)
            # First patch should be evicted by LRU pruning and cascade-deleted from binary_frames
            self.assertIsNone(tiny_cache.get_manifest(grid_a))
            self.assertIsNone(tiny_cache.get_frame(grid_a, "pass_1.bin"))
            with tiny_cache.connection() as conn:
                remaining_frames = conn.execute(
                    "SELECT COUNT(*) FROM binary_frames WHERE grid_key = ?",
                    (grid_a,),
                ).fetchone()[0]
            self.assertEqual(int(remaining_frames), 0)
            self.assertIsNotNone(tiny_cache.get_manifest(m_b["query"]["grid_key"]))
        finally:
            tiny_cache.close()

    def test_sqlite_connection_pooling_and_zero_resource_warnings_on_gc(self) -> None:
        import gc
        import threading
        import warnings

        pool_db = str(Path(self.temp_dir.name) / "pool.db")

        def _exercise_pool() -> None:
            pooled_cache = SpatialSqliteCache(db_path=pool_db)
            with pooled_cache.connection() as c1:
                conn_main_1 = c1
            with pooled_cache.connection() as c2:
                conn_main_2 = c2
            self.assertIs(conn_main_1, conn_main_2)

            worker_conns: list[object] = []

            def _worker() -> None:
                with pooled_cache.connection() as wc:
                    worker_conns.append(wc)
                    wc.execute("SELECT 1").fetchone()

            t = threading.Thread(target=_worker)
            t.start()
            t.join()
            self.assertEqual(len(worker_conns), 1)
            self.assertIsNot(worker_conns[0], conn_main_1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            _exercise_pool()
            gc.collect()

        resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(resource_warnings, [])

    def test_forced_spectrum_extracts_102_wavelength_planes(self) -> None:
        ra, dec = 133.7951, -7.2451
        cutouts = [
            self._make_spectral_cutout(
                ra,
                dec,
                bandpass=f"SPHEREx-D{(index // 17) + 1}",
                wavelength=float(wavelength),
                source_flux=5.0,
            )
            for index, wavelength in enumerate(np.linspace(0.75, 5.0, 102))
        ]

        payload = extract_forced_spectrum(ra, dec, list(reversed(cutouts)), "Test source")
        parsed = validate_spectrum_payload(payload)
        self.assertEqual(parsed.spectrum_origin, "SPHEREx 102-Band Forced Photometry")
        self.assertEqual(len(parsed.channels), 102)
        self.assertAlmostEqual(parsed.channels[0].wavelength_um, 0.75)
        self.assertAlmostEqual(parsed.channels[-1].wavelength_um, 5.0)
        self.assertTrue(all(np.isfinite(channel.flux) for channel in parsed.channels))
        self.assertTrue(
            all(
                np.isfinite(channel.variance) and channel.variance >= 0
                for channel in parsed.channels
            )
        )
        self.assertEqual(len(parsed.molecular_features), 4)
        self.assertTrue(all(channel.flux > 0 for channel in parsed.channels))

    def test_forced_spectrum_uses_detector_bands_when_sub_exposures_unavailable(self) -> None:
        ra, dec = 133.7951, -7.2451
        cutouts = [
            self._make_spectral_cutout(
                ra,
                dec,
                bandpass=f"SPHEREx-D{detector}",
                source_flux=5.0 + detector,
            )
            for detector in range(1, 7)
        ]

        parsed = validate_spectrum_payload(
            extract_forced_spectrum(ra, dec, cutouts, "Six detector bands")
        )
        self.assertEqual(
            parsed.spectrum_origin,
            "SPHEREx Detector-Band Forced Photometry (Interpolated 102-Grid)",
        )
        self.assertEqual(len(parsed.channels), 102)
        self.assertAlmostEqual(parsed.channels[0].wavelength_um, 0.75)
        self.assertAlmostEqual(parsed.channels[-1].wavelength_um, 5.0)
        self.assertEqual({channel.detector for channel in parsed.channels}, set(range(1, 7)))
        self.assertTrue(all(np.isfinite(channel.flux) for channel in parsed.channels))
        self.assertTrue(all(channel.variance >= 0 for channel in parsed.channels))
        self.assertEqual(
            [feature.molecule for feature in parsed.molecular_features],
            ["H2O", "CH4", "CO2", "CO"],
        )

    def test_forced_spectrum_service_and_api_return_102_channels(self) -> None:
        from fastapi.testclient import TestClient

        ra, dec = 133.7951, -7.2451
        cutouts = [
            self._make_spectral_cutout(
                ra,
                dec,
                bandpass=f"SPHEREx-D{detector}",
                source_flux=5.0 + detector,
            )
            for detector in range(1, 7)
        ]

        service = StarHuntersBackendService(
            cache=self.cache,
            spherex_cutout_fetcher=lambda **_kwargs: cutouts,
        )
        service_payload = service.get_forced_spectrum(ra, dec)
        client = TestClient(create_app(service))
        response = client.get(
            "/api/v1/spectrum",
            params={"ra": ra, "dec": dec},
        )
        labeled_response = client.get(
            "/api/v1/spectrum",
            params={"ra": ra, "dec": dec, "label": "WISE J0855-0714"},
        )
        unlabeled_after_labeled = client.get(
            "/api/v1/spectrum",
            params={"ra": ra, "dec": dec},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_label"], "Selected sky position")
        self.assertEqual(labeled_response.status_code, 200)
        self.assertEqual(labeled_response.json()["source_label"], "WISE J0855-0714")
        self.assertEqual(unlabeled_after_labeled.status_code, 200)
        self.assertEqual(unlabeled_after_labeled.json()["source_label"], "Selected sky position")
        for payload in (service_payload, response.json(), labeled_response.json()):
            parsed = validate_spectrum_payload(payload)
            self.assertEqual(len(parsed.channels), 102)
            self.assertEqual(len(parsed.molecular_features), 4)

    def test_forced_spectrum_offline_synthetic_fallback_returns_102_channels(self) -> None:
        with patch.dict("os.environ", {"SPHEREX_OFFLINE": "1"}):
            payload = self.service.get_forced_spectrum(133.7951, -7.2451)

        parsed = validate_spectrum_payload(payload)
        self.assertEqual(len(parsed.channels), 102)
        self.assertAlmostEqual(parsed.channels[0].wavelength_um, 0.75)
        self.assertAlmostEqual(parsed.channels[-1].wavelength_um, 5.0)
        self.assertEqual({channel.detector for channel in parsed.channels}, set(range(1, 7)))
        self.assertEqual(len(parsed.molecular_features), 4)

    def test_forced_spectrum_rejects_incomplete_wavelength_set(self) -> None:
        with self.assertRaises(SpectrumExtractionError):
            extract_forced_spectrum(133.7951, -7.2451, [], "Incomplete")

    def test_spectrum_102_channels_and_monotonic_order(self) -> None:
        spec = build_spectrum_payload(133.7951, -7.2451, "WISE 0855", slug="wise_0855")
        parsed = validate_spectrum_payload(spec)
        self.assertEqual(len(parsed.channels), 102)
        self.assertEqual(len(parsed.molecular_features), 4)

        # Verify that each showcase target has a distinct 102-band spectral profile
        signatures = set()
        for slug, meta in SHOWCASE_TARGETS.items():
            s = build_spectrum_payload(meta["ra"], meta["dec"], meta["name"], slug=slug)
            validate_spectrum_payload(s)
            flux_tuple = tuple(ch["flux"] for ch in s["channels"][:10])
            signatures.add(flux_tuple)
        self.assertEqual(len(signatures), len(SHOWCASE_TARGETS))

        # Verify SQLite spectrum caching round-trip via backend_service.get_spectrum
        s1 = self.service.get_spectrum(133.7951, -7.2451)
        s2 = self.service.get_spectrum(133.7951, -7.2451)
        self.assertEqual(s1["channels"][0]["flux"], s2["channels"][0]["flux"])

        # Reject non-monotonic channels
        bad_spec = copy.deepcopy(spec)
        bad_spec["channels"][10]["wavelength_um"] = 0.70
        with self.assertRaises(ValueError):
            validate_spectrum_payload(bad_spec)

    def test_earth_to_sky_zenith_bridge_random_capital_fallback_and_sqlite_cache(self) -> None:
        res1 = compute_zenith_icrs(lat=-0.0917, lon=34.7680, iso_time="2026-11-14T16:21:00Z")
        res2 = compute_zenith_icrs(lat=-0.0917, lon=49.7680, iso_time="2026-11-14T16:21:00Z")
        validate_earth_to_sky_response(res1)
        validate_earth_to_sky_response(res2)
        self.assertFalse(res1["earth"]["is_fallback_capital"])
        # +15 deg longitude at same UTC shifts Zenith RA by +15 deg
        dra = (res2["zenith_icrs"]["ra"] - res1["zenith_icrs"]["ra"]) % 360.0
        self.assertAlmostEqual(dra, 15.0, places=2)

        # When user location data is unavailable (lat/lon omitted), pick a random capital city
        # and cache both the Zenith lookup and the overhead sky patch in SQLite.
        fb1 = self.service.get_earth_to_sky(
            lat=None,
            lon=None,
            iso_time="2026-11-14T16:21:00Z",
            fallback_seed=7,
            prewarm_patch=True,
        )
        validate_earth_to_sky_response(fb1)
        self.assertTrue(fb1["earth"]["is_fallback_capital"])
        self.assertFalse(fb1["earth"]["cached"])
        self.assertIsNotNone(fb1["earth"]["city_name"])
        self.assertIsNotNone(fb1["earth"]["country"])
        capital_names = {c["city_name"] for c in WORLD_CAPITAL_CITIES}
        self.assertGreater(len(WORLD_CAPITAL_CITIES), 0)
        self.assertEqual(len({c["slug"] for c in WORLD_CAPITAL_CITIES}), len(WORLD_CAPITAL_CITIES))
        self.assertIn(fb1["earth"]["city_name"], capital_names)

        # Repeat lookup hits SQLite capital_zenith_cache (cached=True)
        fb2 = self.service.get_earth_to_sky(
            lat=None,
            lon=None,
            iso_time="2026-11-14T16:21:00Z",
            fallback_seed=7,
            prewarm_patch=True,
        )
        self.assertTrue(fb2["earth"]["cached"])
        self.assertEqual(fb1["earth"]["city_name"], fb2["earth"]["city_name"])

        # And the overhead sky patch for that capital city was pre-warmed in SQLite
        cap_grid_key, _, _ = compute_grid_key(
            fb1["zenith_icrs"]["ra"], fb1["zenith_icrs"]["dec"], 5.0
        )
        cached_patch = self.cache.get_manifest(cap_grid_key)
        self.assertIsNotNone(cached_patch)
        self.assertTrue(cached_patch["query"]["cached"])

        # Regression test: when fallback_seed is None (default production path),
        # earth coordinates and zenith_icrs coordinates must always match the SAME capital city.
        for i in range(15):
            unseeded_fb = self.service.get_earth_to_sky(
                lat=None,
                lon=None,
                iso_time=f"2026-11-14T16:{i:02d}:00Z",
                fallback_seed=None,
                prewarm_patch=False,
            )
            validate_earth_to_sky_response(unseeded_fb)
            self.assertTrue(unseeded_fb["earth"]["is_fallback_capital"])
            self.assertAlmostEqual(
                unseeded_fb["zenith_icrs"]["dec"],
                unseeded_fb["earth"]["lat"],
                delta=1.0,
            )
            expected_zenith = compute_zenith_icrs(
                lat=unseeded_fb["earth"]["lat"],
                lon=unseeded_fb["earth"]["lon"],
                iso_time=unseeded_fb["earth"]["iso_time"],
            )
            self.assertAlmostEqual(
                unseeded_fb["zenith_icrs"]["ra"],
                expected_zenith["zenith_icrs"]["ra"],
                places=4,
            )
            self.assertAlmostEqual(
                unseeded_fb["zenith_icrs"]["dec"],
                expected_zenith["zenith_icrs"]["dec"],
                places=4,
            )
            self.assertEqual(
                unseeded_fb["zenith_icrs"]["constellation"],
                expected_zenith["zenith_icrs"]["constellation"],
            )

        # Verify GMST/LST sidereal fallback agrees within ~0.5 deg of astropy ICRS
        gmst_fb = compute_zenith_icrs(
            lat=-0.0917, lon=34.7680, iso_time="2026-11-14T16:21:00Z", use_astropy=False
        )
        validate_earth_to_sky_response(gmst_fb)
        self.assertLess(abs(gmst_fb["zenith_icrs"]["dec"] - res1["zenith_icrs"]["dec"]), 0.5)

        with self.assertRaises(ValueError):
            compute_zenith_icrs(lat=95.0, lon=0.0)
        with self.assertRaises(ValueError):
            compute_zenith_icrs(lat=0.0, lon=195.0)

        # HTTP endpoint validation: out-of-range lat/lon returns 400 Bad Request
        from fastapi.testclient import TestClient

        client = TestClient(create_app(self.service))
        self.assertEqual(
            client.get("/api/v1/earth-to-sky", params={"lat": 95.0, "lon": 0.0}).status_code,
            400,
        )
        self.assertEqual(
            client.get("/api/v1/earth-to-sky", params={"lat": 0.0, "lon": -185.0}).status_code,
            400,
        )
        ok_resp = client.get("/api/v1/earth-to-sky")
        self.assertEqual(ok_resp.status_code, 200)
        ok_payload = ok_resp.json()
        validate_earth_to_sky_response(ok_payload)
        self.assertTrue(ok_payload["earth"]["is_fallback_capital"])
        self.assertTrue(ok_payload["earth"]["city_name"])
        self.assertTrue(ok_payload["earth"]["country"])

        # Verify time alias fallback when iso_time is omitted, and iso_time precedence
        alias_resp = client.get(
            "/api/v1/earth-to-sky",
            params={"lat": -0.0917, "lon": 34.7680, "time": "2026-11-14T16:21:00Z"},
        )
        self.assertEqual(alias_resp.status_code, 200)
        self.assertEqual(alias_resp.json()["earth"]["iso_time"], "2026-11-14T16:21:00Z")

        precedence_resp = client.get(
            "/api/v1/earth-to-sky",
            params={
                "lat": -0.0917,
                "lon": 34.7680,
                "iso_time": "2026-11-14T16:21:00Z",
                "time": "2026-01-01T00:00:00Z",
            },
        )
        self.assertEqual(precedence_resp.status_code, 200)
        self.assertEqual(precedence_resp.json()["earth"]["iso_time"], "2026-11-14T16:21:00Z")

    def test_highlighter_palette_fuzzy_search_and_pinned_cache(self) -> None:
        # At least 16 distinct highlighter colors
        self.assertGreaterEqual(len(TRACKING_SQUARE_COLORS), 16)
        self.assertEqual(len(set(TRACKING_SQUARE_COLORS)), len(TRACKING_SQUARE_COLORS))

        # Fuzzy search finds objects by exact name, partial token, subsequence, or typo
        barnard_hits = self.service.search_known_objects(query="barnrd", limit=10)
        self.assertGreater(len(barnard_hits), 0)
        self.assertEqual(barnard_hits[0]["target_slug"], "barnards_star")

        luhman_hits = self.service.search_known_objects(query="luhman", limit=10)
        self.assertGreater(len(luhman_hits), 0)
        self.assertEqual(luhman_hits[0]["target_slug"], "luhman_16")

        ceres_hits = self.service.search_known_objects(query="ceres", limit=10)
        self.assertGreater(len(ceres_hits), 0)
        self.assertEqual(ceres_hits[0]["target_slug"], "ceres_asteroid")

        # Segmented SLRU eviction: 2-visitor shared patch survives eviction ahead of 1-visitor patch
        slru_db = str(Path(self.temp_dir.name) / "slru.db")
        slru_cache = SpatialSqliteCache(db_path=slru_db, max_cache_bytes=20_000_000)
        svc = StarHuntersBackendService(cache=slru_cache)

        # User A creates patch 1, User B visits it (2 distinct visitors),
        # and User A creates patch 2 (1 visitor).
        m_shared = svc.get_sky_patch(
            133.7951, -7.2451, 5.0, client_ip="10.0.0.1", user_agent="UA-A"
        )
        svc.get_sky_patch(133.7951, -7.2451, 5.0, client_ip="10.0.0.2", user_agent="UA-B")
        m_single = svc.get_sky_patch(15.0, 15.0, 5.0, client_ip="10.0.0.1", user_agent="UA-A")
        shared_size = slru_cache.get_patch_byte_size(m_shared["query"]["grid_key"])
        single_size = slru_cache.get_patch_byte_size(m_single["query"]["grid_key"])
        slru_cache.max_cache_bytes = int(shared_size + single_size * 1.15)

        # Inserting a third patch must evict the 1-visitor patch, preserving the 2-visitor patch
        m_single2 = svc.get_sky_patch(45.0, 25.0, 5.0, client_ip="10.0.0.3", user_agent="UA-C")
        self.assertIsNotNone(slru_cache.get_manifest(m_shared["query"]["grid_key"]))
        self.assertIsNone(slru_cache.get_manifest(m_single["query"]["grid_key"]))
        self.assertIsNotNone(slru_cache.get_manifest(m_single2["query"]["grid_key"]))

    def test_fastapi_progressive_endpoints_gzip_binary_streaming_and_admin_clear(self) -> None:
        from fastapi.middleware.gzip import GZipMiddleware
        from fastapi.testclient import TestClient

        test_app = create_app(self.service)

        # Verify GZipMiddleware is configured on the FastAPI app
        self.assertTrue(
            any(m.cls is GZipMiddleware for m in test_app.user_middleware),
            "GZipMiddleware must be registered on FastAPI app",
        )

        client = TestClient(test_app)

        # 1. /api/v1/catalogs
        cat_resp = client.get(
            "/api/v1/catalogs",
            params={"ra": 133.7951, "dec": -7.2451, "fov": 5.0, "live_catalogs": False},
        )
        self.assertEqual(cat_resp.status_code, 200)
        self.assertEqual(cat_resp.headers.get("cache-control"), "public, max-age=3600")
        validate_catalogs_response(cat_resp.json())

        # 2. /api/v1/sky-patch
        patch_resp = client.get(
            "/api/v1/sky-patch",
            params={"ra": 133.7951, "dec": -7.2451, "fov": 5.0, "live_catalogs": False},
        )
        self.assertEqual(patch_resp.status_code, 200)
        self.assertEqual(patch_resp.headers.get("cache-control"), "public, max-age=3600")
        manifest = patch_resp.json()
        validate_sky_patch_manifest(manifest)
        grid_key = manifest["query"]["grid_key"]

        # 3. /api/v1/frames/{grid_key}/pass_1.bin?precision=float16 (786,432 bytes)
        f16_resp = client.get(
            f"/api/v1/frames/{grid_key}/pass_1.bin",
            params={"precision": "float16"},
        )
        self.assertEqual(f16_resp.status_code, 200)
        self.assertEqual(len(f16_resp.content), 786_432)
        self.assertEqual(f16_resp.headers.get("x-binary-dtype"), "float16")
        self.assertEqual(
            f16_resp.headers.get("cache-control"), "public, max-age=31536000, immutable"
        )

        # 4. /api/v1/frames/{grid_key}/pass_1.bin?precision=float32 (1,572,864 bytes)
        f32_resp = client.get(
            f"/api/v1/frames/{grid_key}/pass_1.bin",
            params={"precision": "float32"},
        )
        self.assertEqual(f32_resp.status_code, 200)
        self.assertEqual(len(f32_resp.content), 1_572_864)
        self.assertEqual(f32_resp.headers.get("x-binary-dtype"), "float32")
        self.assertEqual(
            f32_resp.headers.get("cache-control"), "public, max-age=31536000, immutable"
        )

        # 5. /api/v1/frames/{grid_key}/pass_1_flags.bin (uint8)
        flags_resp = client.get(f"/api/v1/frames/{grid_key}/pass_1_flags.bin")
        self.assertEqual(flags_resp.status_code, 200)
        self.assertEqual(len(flags_resp.content), BINARY_FLAGS_UINT8_BYTES)
        self.assertEqual(flags_resp.headers.get("x-binary-dtype"), "uint8")

        # 6. /api/v1/earth-to-sky
        e2s_resp = client.get("/api/v1/earth-to-sky")
        self.assertEqual(e2s_resp.status_code, 200)
        validate_earth_to_sky_response(e2s_resp.json())

        # 7. POST /api/v1/admin/clear-cache (403 without valid token, 200 with valid token)
        self.assertEqual(client.post("/api/v1/admin/clear-cache").status_code, 403)
        self.assertEqual(
            client.post(
                "/api/v1/admin/clear-cache",
                headers={"X-Admin-Token": "wrong-token"},
            ).status_code,
            403,
        )
        clear_resp = client.post(
            "/api/v1/admin/clear-cache",
            headers={"X-Admin-Token": self.service.admin_token},
        )
        self.assertEqual(clear_resp.status_code, 200)
        self.assertEqual(clear_resp.json(), {"status": "cleared"})
        self.assertIsNone(self.cache.get_manifest(grid_key))

    def test_dockerfile_fly_toml_and_unified_spa_serving(self) -> None:
        import tomllib

        repo_root = Path(__file__).resolve().parent.parent.parent
        dockerfile_text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM node:22-alpine AS frontend-builder", dockerfile_text)
        self.assertIn("FROM python:3.11-slim AS runtime", dockerfile_text)
        self.assertIn(
            "COPY --from=frontend-builder /app/frontend/dist ./frontend/dist", dockerfile_text
        )
        self.assertIn("SQLITE_DB_PATH=/data/spherex_cache.db", dockerfile_text)

        fly_cfg = tomllib.loads((repo_root / "fly.toml").read_text(encoding="utf-8"))
        self.assertEqual(fly_cfg["primary_region"], "iad")
        self.assertEqual(fly_cfg["env"]["SQLITE_DB_PATH"], "/data/spherex_cache.db")
        self.assertEqual(fly_cfg["env"]["IN_MEMORY_CACHE_MB"], "512")
        self.assertEqual(fly_cfg["env"]["SQLITE_MAX_CACHE_MB"], "1536")
        self.assertEqual(fly_cfg["http_service"]["auto_stop_machines"], "suspend")
        self.assertEqual(fly_cfg["http_service"]["min_machines_running"], 0)
        self.assertEqual(fly_cfg["http_service"]["concurrency"]["type"], "connections")
        self.assertEqual(fly_cfg["http_service"]["concurrency"]["soft_limit"], 30)
        self.assertEqual(fly_cfg["http_service"]["concurrency"]["hard_limit"], 50)
        self.assertEqual(fly_cfg["mounts"][0]["destination"], "/data")
        self.assertEqual(fly_cfg["mounts"][0]["initial_size"], "2gb")
        self.assertEqual(fly_cfg["vm"][0]["size"], "shared-cpu-2x")
        self.assertEqual(fly_cfg["vm"][0]["memory"], "2048mb")

        frontend_dist = repo_root / "frontend" / "dist"
        if (frontend_dist / "index.html").is_file():
            from fastapi.testclient import TestClient

            self.assertIsNotNone(app)
            with TestClient(app) as client:
                root_resp = client.get("/")
                self.assertEqual(root_resp.status_code, 200)
                self.assertIn("text/html", root_resp.headers.get("content-type", ""))
                self.assertEqual(client.get("/health").status_code, 200)

    def test_forced_spectrum_skips_fully_flagged_cutout_when_other_cutouts_are_valid(self) -> None:
        ra, dec = 133.7951, -7.2451
        valid_cutouts = [
            self._make_spectral_cutout(
                ra,
                dec,
                bandpass=f"SPHEREx-D{detector}",
                source_flux=5.0 + detector,
            )
            for detector in range(1, 7)
        ]
        bad_cutout = self._make_spectral_cutout(ra, dec, bandpass="SPHEREx-D1", source_flux=5.0)
        bad_cutout.flags_mask.fill(1)

        parsed = validate_spectrum_payload(
            extract_forced_spectrum(ra, dec, [bad_cutout, *valid_cutouts], "Partial bad cutout")
        )
        self.assertEqual(len(parsed.channels), 102)
        self.assertEqual(
            parsed.spectrum_origin,
            "SPHEREx Detector-Band Forced Photometry (Interpolated 102-Grid)",
        )

    def test_object_epochs_clustering_and_api_endpoint(self) -> None:
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        from app.utils.astronomy import datetime_to_mjd

        # 1. cluster_most_matches_into_passes groups exposures >15d apart & picks median MJD coords
        clustered = cluster_most_matches_into_passes(
            [
                {"ra": 1.1, "dec": -1.1, "mjd_start": 60841.0},
                {"ra": 1.2, "dec": -1.2, "mjd_start": 60842.0},
                {"ra": 1.3, "dec": -1.3, "mjd_start": 60843.0},
                {"ra": 359.4, "dec": -2.4, "mjd_start": 61020.0},
            ]
        )
        self.assertEqual(len(clustered), 2)
        self.assertEqual(clustered[0]["epoch_key"], "spherex_pass_1")
        self.assertEqual(clustered[0]["ra"], 1.2)
        self.assertEqual(clustered[0]["exposure_count"], 3)
        self.assertEqual(clustered[1]["epoch_key"], "spherex_pass_2")
        self.assertEqual(clustered[1]["ra"], 359.4)

        now_mjd = datetime_to_mjd()
        self.cache.upsert_ephemeris_points(
            [
                {
                    "naif_id": "699",
                    "body_name": "Saturn",
                    "body_class": "planet",
                    "mjd": now_mjd,
                    "ra": 355.1234,
                    "dec": -3.4567,
                    "vmag": 0.8,
                    "source": "jpl_horizons_test",
                }
            ]
        )

        async def _fake_most(obj_name: str) -> dict[str, object]:
            if obj_name.lower() == "saturn":
                return {
                    "query_name": "Saturn",
                    "status": "ok",
                    "matches": [
                        {"ra": 1.7387, "dec": -1.6102, "mjd_start": 60841.77},
                        {"ra": 356.2, "dec": -3.1, "mjd_start": 61005.25},
                    ],
                }
            return {"query_name": obj_name, "status": "no_match", "matches": []}

        client = TestClient(create_app(self.service))
        with patch.object(
            self.service, "lookup_most_object", new=AsyncMock(side_effect=_fake_most)
        ):
            saturn_resp = client.get(
                "/api/v1/catalogs/object-epochs",
                params={
                    "object_id": "horizons:699",
                    "label": "Saturn",
                    "object_type": "solar_system",
                    "object_class": "planet",
                    "ra": 355.8,
                    "dec": -2.9,
                },
            )
            self.assertEqual(saturn_resp.status_code, 200)
            sat_data = saturn_resp.json()
            self.assertTrue(sat_data["is_solar_system"])
            self.assertEqual(sat_data["passes_status"], "ok")
            self.assertEqual(
                [e["kind"] for e in sat_data["epochs"]],
                ["current", "spherex_pass", "spherex_pass"],
            )
            self.assertFalse(sat_data["epochs"][0]["can_view_in_blinker"])
            self.assertAlmostEqual(sat_data["epochs"][0]["ra"], 355.1234, places=3)
            self.assertTrue(sat_data["epochs"][1]["can_view_in_blinker"])
            self.assertAlmostEqual(sat_data["epochs"][1]["ra"], 1.7387, places=4)

            mars_resp = client.get(
                "/api/v1/catalogs/object-epochs",
                params={
                    "object_id": "horizons:499",
                    "label": "Mars",
                    "object_type": "solar_system",
                    "object_class": "planet",
                    "ra": 115.2,
                    "dec": 23.4,
                },
            )
            self.assertEqual(mars_resp.status_code, 200)
            mars_data = mars_resp.json()
            self.assertEqual(mars_data["passes_status"], "no_spherex_passes")
            self.assertEqual([e["kind"] for e in mars_data["epochs"]], ["current"])

        wise_resp = client.get(
            "/api/v1/catalogs/object-epochs",
            params={
                "object_id": "wise:J085510.83-071442.5",
                "label": "WISE 0855-0714",
                "object_type": "brown_dwarf_candidate",
                "target_slug": "wise_0855",
                "ra": 133.7951,
                "dec": -7.2451,
            },
        )
        self.assertEqual(wise_resp.status_code, 200)
        wise_data = wise_resp.json()
        self.assertFalse(wise_data["is_solar_system"])
        self.assertEqual(wise_data["epochs"][0]["kind"], "current")
        self.assertTrue(wise_data["epochs"][0]["can_view_in_blinker"])
        self.assertEqual(wise_data["passes_status"], "ok")
        self.assertGreaterEqual(
            sum(1 for ep in wise_data["epochs"] if ep["kind"] == "spherex_pass"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
