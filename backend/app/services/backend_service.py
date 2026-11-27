"""Core backend service orchestrating spatial caching, catalogs, frames, and spectra."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gzip
import math
import os
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple

from app.models.contracts import (
    SHOWCASE_TARGETS,
    convert_float32_bytes_to_float16_bytes,
    validate_catalogs_response,
    validate_earth_to_sky_response,
    validate_sky_patch_manifest,
    validate_spectrum_payload,
)
from app.services.archival_resolver import (
    ArchivalResolver,
    apply_solar_system_ephemeris_to_frames,
    merge_pipeline_overlays,
)
from app.services.catalog_clients import fetch_most_spherex_matches
from app.services.celestial_catalog_pipeline import AutomatedCelestialPipeline
from app.services.logging_config import configure_backend_logging, get_logger
from app.services.object_epochs import resolve_object_observation_epochs
from app.services.object_search import (
    get_global_known_objects_index,
    search_known_objects,
    search_known_objects_async,
)
from app.services.skyview_fallback import fetch_skyview_fallback
from app.services.spatial_cache import SpatialSqliteCache, compute_grid_key
from app.services.spectrum_extractor import extract_forced_spectrum
from app.services.spherex_coadd import build_target_wcs_header, coadd_spherex_epochs
from app.services.spherex_ingest import (
    DEFAULT_MAX_SPECTRUM_CUTOUTS,
    _env_float,
    ingest_spherex_cutouts,
)
from app.services.synthetic_patch import (
    build_sky_patch_manifest_payload,
    build_spectrum_payload,
    generate_synthetic_epoch_frames,
)
from app.services.wise_fallback import fetch_wise_fallback
from app.services.zenith_bridge import (
    compute_zenith_icrs,
    find_matching_capital_city,
    pick_random_capital_city,
)
from app.utils.astronomy import datetime_to_mjd

_logger = get_logger("backend")

DEFAULT_COLD_MISS_RATE_LIMIT_PER_MIN = 30
CPU_PRIORITY_SPECTRUM = 0
CPU_PRIORITY_SKY_PATCH = 10
DEFAULT_CPU_STARVATION_THRESHOLD_S = 2.0


class _WaiterEntry(NamedTuple):
    """Queued waiter entry for ``_AgingPrioritySemaphore``.

    Attributes:
        priority: Caller priority (lower values are dispatched first).
        enqueued_at: Monotonic timestamp when the waiter was queued, used for
            aging/starvation promotion and oldest-first ordering.
        seq: Monotonically increasing sequence number for deterministic FIFO
            tie-breaking when timestamps match.
        future: Future resolved by ``release()`` when this waiter wins a slot.
    """

    priority: int
    enqueued_at: float
    seq: int
    future: asyncio.Future[None]


class _AgingPrioritySemaphore:
    """Async semaphore with priority ordering and anti-starvation aging."""

    def __init__(
        self,
        capacity: int,
        starvation_threshold_s: float = DEFAULT_CPU_STARVATION_THRESHOLD_S,
    ) -> None:
        self._capacity = max(1, int(capacity))
        self._starvation_threshold_s = max(0.01, float(starvation_threshold_s))
        self._in_use = 0
        self._seq = 0
        self._waiters: list[_WaiterEntry] = []

    def _prune_done_waiters(self) -> None:
        if self._waiters:
            self._waiters = [w for w in self._waiters if not w.future.done()]

    async def acquire(self, priority: int) -> None:
        """Acquire a CPU slot, queueing by priority when saturated."""
        self._prune_done_waiters()
        if self._in_use < self._capacity and not self._waiters:
            self._in_use += 1
            return

        fut = asyncio.get_running_loop().create_future()
        self._seq += 1
        entry = _WaiterEntry(
            priority=priority,
            enqueued_at=time.monotonic(),
            seq=self._seq,
            future=fut,
        )
        self._waiters.append(entry)
        try:
            await fut
        except BaseException:
            if entry in self._waiters:
                self._waiters.remove(entry)
            if fut.done() and not fut.cancelled():
                self.release()
            raise

    def release(self) -> None:
        """Release a slot, promoting any waiter past ``starvation_threshold_s`` first."""
        self._prune_done_waiters()
        if not self._waiters:
            self._in_use = max(0, self._in_use - 1)
            return

        now = time.monotonic()
        best_idx = 0
        # Sort key: (starvation_tier [0=starved, 1=normal], effective_priority, enqueued_at, seq)
        best_key: tuple[int, int, float, int] | None = None
        for idx, waiter in enumerate(self._waiters):
            starved = (now - waiter.enqueued_at) >= self._starvation_threshold_s
            key = (
                (0, 0, waiter.enqueued_at, waiter.seq)
                if starved
                else (1, waiter.priority, waiter.enqueued_at, waiter.seq)
            )
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx

        winner = self._waiters.pop(best_idx)
        winner.future.set_result(None)

    @asynccontextmanager
    async def slot(self, priority: int) -> AsyncIterator[None]:
        """Acquire and deterministically release a prioritized CPU slot."""
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()


class ColdMissRateLimitError(Exception):
    """Raised when a single client IP exceeds the cold archival synthesis rate limit."""

    def __init__(self, retry_after_s: float, limit_per_min: int) -> None:
        self.retry_after_s = max(1.0, float(retry_after_s))
        self.limit_per_min = int(limit_per_min)
        super().__init__(
            f"Cold archival synthesis rate limit exceeded ({self.limit_per_min}/min); "
            f"retry after {math.ceil(self.retry_after_s)}s"
        )


class ColdMissRateLimiter:
    """Thread-safe per-IP 60s sliding-window rate limiter for uncached cold synthesis misses."""

    def __init__(self, limit_per_min: int) -> None:
        self.limit_per_min = max(0, int(limit_per_min))
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, client_ip: str) -> tuple[bool, float]:
        """Record a cold miss if under quota; return ``(allowed, retry_after_s)``."""
        if self.limit_per_min <= 0:
            return True, 0.0
        now = time.monotonic()
        cutoff = now - 60.0
        key = client_ip or "unknown"
        with self._lock:
            if len(self._events) > 4096:
                stale_ips = [ip for ip, dq in self._events.items() if not dq or dq[-1] <= cutoff]
                for ip in stale_ips:
                    self._events.pop(ip, None)
            dq = self._events.setdefault(key, deque())
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.limit_per_min:
                retry_after = max(1.0, 60.0 - (now - dq[0]))
                return False, retry_after
            dq.append(now)
            return True, 0.0


def compute_max_concurrent_coadds(cpu_cores: int | None = None) -> int:
    """Compute dynamic max concurrent archival co-add workers (~2/3 of available CPU cores).

    Reserves ``max(1, total // 3)`` cores for ``asyncio`` + ``_io_executor`` so
    I/O always retains at least 1 core while giving ~2/3 of cores to CPU coadds.

    Args:
        cpu_cores: Optional explicit core count. Defaults to affinity-visible cores
            (``os.sched_getaffinity(0)``) or ``os.cpu_count()``.

    Returns:
        Maximum concurrent archival co-add worker threads (at least ``1``).
    """
    if cpu_cores is None:
        if hasattr(os, "sched_getaffinity"):
            try:
                cpu_cores = len(os.sched_getaffinity(0))
            except OSError:
                cpu_cores = os.cpu_count() or 2
        else:
            cpu_cores = os.cpu_count() or 2
    total = max(1, int(cpu_cores))
    reserved_for_io = max(1, total // 3)
    return max(1, total - reserved_for_io)


class StarHuntersBackendService:
    """Core request handler used by both FastAPI routes and standalone tests."""

    def __init__(
        self,
        cache: SpatialSqliteCache | None = None,
        enable_archival_data: bool | None = None,
        wise_fallback_fetcher: Callable[..., Any] = fetch_wise_fallback,
        skyview_fallback_fetcher: Callable[..., Any] = fetch_skyview_fallback,
        spherex_cutout_fetcher: Callable[..., list[Any]] = ingest_spherex_cutouts,
        spherex_coadder: Callable[..., Any] = coadd_spherex_epochs,
        wcs_header_builder: Callable[..., dict[str, Any]] = build_target_wcs_header,
        spectrum_extractor: Callable[..., dict[str, Any]] = extract_forced_spectrum,
        catalog_pipeline: AutomatedCelestialPipeline | None = None,
        archival_resolver: ArchivalResolver | None = None,
        cold_miss_rate_limit_per_min: int | None = None,
    ) -> None:
        """Initialize the backend service with optional injected dependencies.

        Args:
            cache: SQLite spatial cache instance (defaults to a new
                :class:`SpatialSqliteCache`).
            enable_archival_data: Whether live archival IRSA/WISE/SkyView queries
                are enabled; defaults to ``ENABLE_ARCHIVAL_DATA`` env var.
            wise_fallback_fetcher: Callable fetching WISE/NEOWISE fallback epochs.
            skyview_fallback_fetcher: Callable fetching NASA SkyView fallback epochs.
            spherex_cutout_fetcher: Callable ingesting SPHEREx cutouts.
            spherex_coadder: Callable clustering and co-adding SPHEREx cutouts.
            wcs_header_builder: Callable constructing FITS TAN WCS headers.
            spectrum_extractor: Callable extracting 102-band forced spectra.
            catalog_pipeline: Optional pre-configured :class:`AutomatedCelestialPipeline`.
            archival_resolver: Optional pre-configured :class:`ArchivalResolver`.
        """
        configure_backend_logging()
        self.cache = cache or SpatialSqliteCache()
        self.admin_token = os.environ.get("ADMIN_TOKEN", "star-hunters-dev-token")
        self.catalog_pipeline = catalog_pipeline or AutomatedCelestialPipeline(cache=self.cache)
        if enable_archival_data is None:
            configured = os.environ.get("ENABLE_ARCHIVAL_DATA", "false").strip().lower()
            enable_archival_data = configured in {"1", "true", "yes", "on"}
        self.enable_archival_data = enable_archival_data
        self.wise_fallback_fetcher = wise_fallback_fetcher
        self.skyview_fallback_fetcher = skyview_fallback_fetcher
        self.spherex_cutout_fetcher = spherex_cutout_fetcher
        self.spherex_coadder = spherex_coadder
        self.wcs_header_builder = wcs_header_builder
        self.spectrum_extractor = spectrum_extractor
        self.archival_resolver = archival_resolver or ArchivalResolver(
            spherex_cutout_fetcher=spherex_cutout_fetcher,
            spherex_coadder=spherex_coadder,
            wcs_header_builder=wcs_header_builder,
            wise_fallback_fetcher=wise_fallback_fetcher,
            skyview_fallback_fetcher=skyview_fallback_fetcher,
        )
        self._inflight_tasks: dict[str, asyncio.Future[Any]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        raw_max_coadds = os.environ.get("ARCHIVAL_MAX_CONCURRENT_COADDS", "").strip()
        max_coadds = (
            max(1, int(raw_max_coadds)) if raw_max_coadds else compute_max_concurrent_coadds()
        )
        self._max_concurrent_coadds = max_coadds
        io_workers = max(8, min(32, (os.cpu_count() or 4) * 4))
        self._cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_coadds,
            thread_name_prefix="spherex-cpu",
        )
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="spherex-io",
        )
        self._executor_finalizer = weakref.finalize(
            self,
            self._shutdown_executors,
            self._cpu_executor,
            self._io_executor,
        )
        self._cpu_gates: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, _AgingPrioritySemaphore
        ] = weakref.WeakKeyDictionary()
        self._archival_semaphore = threading.BoundedSemaphore(max_coadds)

        if cold_miss_rate_limit_per_min is not None:
            cold_limit = max(0, int(cold_miss_rate_limit_per_min))
        else:
            raw_limit = os.environ.get("COLD_MISS_RATE_LIMIT_PER_IP_PER_MIN", "").strip()
            try:
                cold_limit = (
                    max(0, int(raw_limit)) if raw_limit else DEFAULT_COLD_MISS_RATE_LIMIT_PER_MIN
                )
            except ValueError:
                cold_limit = DEFAULT_COLD_MISS_RATE_LIMIT_PER_MIN
        self._cold_miss_limiter = ColdMissRateLimiter(cold_limit)
        _logger.info(
            "StarHuntersBackendService initialized (enable_archival_data=%s, cpu_workers=%d, "
            "io_workers=%d, cold_miss_limit=%d/min)",
            self.enable_archival_data,
            self._max_concurrent_coadds,
            io_workers,
            cold_limit,
        )

    @staticmethod
    def _shutdown_executors(
        cpu_pool: concurrent.futures.ThreadPoolExecutor,
        io_pool: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        """Shutdown worker thread pools cleanly on service garbage collection."""
        for pool in (cpu_pool, io_pool):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _get_cpu_gate(self) -> _AgingPrioritySemaphore:
        """Return an aging priority gate bound to the current running event loop."""
        loop = asyncio.get_running_loop()
        gate = self._cpu_gates.get(loop)
        if gate is None:
            gate = _AgingPrioritySemaphore(self._max_concurrent_coadds)
            self._cpu_gates[loop] = gate
        return gate

    def cancel_background_tasks(self) -> None:
        """Cancel any pending background prewarm tasks."""
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()

    def _schedule_background_patch_prewarm(self, ra: float, dec: float) -> None:
        """Schedule a non-blocking single-flight sky-patch prewarm on the active event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _runner() -> None:
            try:
                await self.get_sky_patch_async(ra=ra, dec=dec, fov=5.0, live_catalogs=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _logger.debug(
                    "Background zenith patch prewarm for (%.2f, %.2f) skipped: %s",
                    ra,
                    dec,
                    exc,
                )

        task = loop.create_task(_runner())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def prewarm_known_data(self) -> int:
        """Pre-warm all showcase manifests, spectra, frames, and the search index.

        Returns:
            Number of showcase targets written to SQLite.
        """
        get_global_known_objects_index()
        count = 0
        for slug, meta in SHOWCASE_TARGETS.items():
            grid_key, ra_q, dec_q = compute_grid_key(meta["ra"], meta["dec"], 5.0)
            manifest = build_sky_patch_manifest_payload(
                ra=meta["ra"],
                dec=meta["dec"],
                fov_arcmin=5.0,
                slug=slug,
            )
            manifest = self._apply_cached_pipeline_sync(
                manifest,
                ra=meta["ra"],
                dec=meta["dec"],
                fov=5.0,
            )
            frames = generate_synthetic_epoch_frames(slug=slug, manifest=manifest)
            self.cache.put_sky_patch(
                grid_key=grid_key,
                ra_center=ra_q,
                dec_center=dec_q,
                fov_arcmin=5.0,
                manifest=manifest,
                frames=frames,
            )
            spec = build_spectrum_payload(
                ra=meta["ra"],
                dec=meta["dec"],
                source_label=meta["name"],
                slug=slug,
            )
            spec_key = f"spec_ra{round(meta['ra'], 4):.4f}_dec{round(meta['dec'], 4):.4f}"
            self.cache.put_spectrum(
                spectrum_key=spec_key,
                ra=meta["ra"],
                dec=meta["dec"],
                payload=spec,
            )
            count += 1
        return count

    def search_known_objects(
        self,
        query: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Fuzzy-search known celestial objects across all indexed targets.

        Args:
            query: Search string.
            limit: Maximum number of results to return.

        Returns:
            Ranked list of matching known-object dictionaries.
        """
        return search_known_objects(query=query, limit=limit)

    async def search_known_objects_async(
        self,
        query: str = "",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Fuzzy-search known objects and refresh live JPL Horizons coordinates.

        Args:
            query: Search string.
            limit: Maximum number of results to return.

        Returns:
            Ranked list of matching known-object dictionaries.
        """
        return await search_known_objects_async(
            query=query,
            limit=limit,
            cache=self.cache,
            catalog_pipeline=self.catalog_pipeline,
            enable_archival_data=self.enable_archival_data,
        )

    async def get_object_observation_epochs_async(
        self,
        *,
        object_id: str = "",
        label: str = "",
        object_type: str | None = None,
        object_class: str | None = None,
        target_slug: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
    ) -> dict[str, Any]:
        """Resolve constrained observation epochs (Current Time + SPHEREx passes only)."""
        return await resolve_object_observation_epochs(
            object_id=object_id,
            label=label,
            object_type=object_type,
            object_class=object_class,
            target_slug=target_slug,
            ra=ra,
            dec=dec,
            cache=self.cache,
            catalog_pipeline=self.catalog_pipeline,
            enable_archival_data=self.enable_archival_data,
            lookup_most_fn=self.lookup_most_object,
        )

    def health(self) -> dict[str, str]:
        """Return service health status payload."""
        return {
            "status": "ok",
            "service": "star-hunters-backend",
            "version": "0.1.0",
        }

    def get_earth_to_sky(
        self,
        lat: float | None = None,
        lon: float | None = None,
        iso_time: str | None = None,
        fallback_seed: int | None = None,
        prewarm_patch: bool = False,
    ) -> dict[str, Any]:
        """Compute or retrieve cached Earth-to-Sky zenith coordinates.

        Args:
            lat: Observer latitude in degrees, or ``None`` for random capital fallback.
            lon: Observer longitude in degrees, or ``None`` for random capital fallback.
            iso_time: Optional ISO-8601 UTC timestamp.
            fallback_seed: Optional deterministic seed for capital city selection.
            prewarm_patch: Whether to populate the overhead sky patch in SQLite.

        Returns:
            Validated ``EarthToSkyResponse`` dictionary.

        Raises:
            ValueError: If ``lat`` or ``lon`` are out of valid geographic bounds.
        """
        is_fallback = lat is None or lon is None
        city_slug: str | None = None
        selected_capital: dict[str, Any] | None = None
        if is_fallback:
            selected_capital = pick_random_capital_city(seed=fallback_seed)
            lat = float(selected_capital["lat"])
            lon = float(selected_capital["lon"])
            city_slug = str(selected_capital["slug"])

        assert lat is not None and lon is not None
        if not (-90.0 <= lat <= 90.0):
            raise ValueError("Latitude must be in [-90.0, 90.0]")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError("Longitude must be in [-180.0, 180.0]")

        if iso_time:
            clean_iso = iso_time.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_iso)
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
            iso_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            dt = datetime.now(UTC)
            iso_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        time_bucket = iso_utc[:16]
        cache_key = (
            f"zenith_{city_slug or 'coord'}_lat{lat:.2f}_lon{lon:.2f}"
            f"_t{time_bucket}_fb{int(is_fallback)}"
        )
        cached_zenith = self.cache.get_zenith_response(cache_key)
        if cached_zenith is not None:
            validate_earth_to_sky_response(cached_zenith)
            if prewarm_patch:
                cached_ra = float(cached_zenith["zenith_icrs"]["ra"])
                cached_dec = float(cached_zenith["zenith_icrs"]["dec"])
                cached_grid_key, _, _ = compute_grid_key(cached_ra, cached_dec, 5.0)
                cached_patch = self.cache.get_manifest(cached_grid_key)
                needs_archival_refresh = (
                    self.enable_archival_data and self._is_pending_archival_demo(cached_patch)
                )
                if cached_patch is None or needs_archival_refresh:
                    self.get_sky_patch(cached_ra, cached_dec, fov=5.0)
            return cached_zenith

        payload = compute_zenith_icrs(
            lat=lat,
            lon=lon,
            iso_time=iso_utc,
        )
        if is_fallback and selected_capital is not None:
            payload["earth"]["lat"] = round(lat, 4)
            payload["earth"]["lon"] = round(lon, 4)
            payload["earth"]["city_name"] = str(selected_capital["city_name"])
            payload["earth"]["country"] = str(selected_capital["country"])
            payload["earth"]["is_fallback_capital"] = True

        self.cache.put_zenith_response(
            cache_key=cache_key,
            lat=payload["earth"]["lat"],
            lon=payload["earth"]["lon"],
            payload=payload,
            city_slug=city_slug,
        )
        if prewarm_patch and (
            is_fallback or city_slug is not None or find_matching_capital_city(lat, lon) is not None
        ):
            self.get_sky_patch(
                ra=payload["zenith_icrs"]["ra"],
                dec=payload["zenith_icrs"]["dec"],
                fov=5.0,
            )
        return payload

    async def get_earth_to_sky_async(
        self,
        lat: float | None = None,
        lon: float | None = None,
        iso_time: str | None = None,
        fallback_seed: int | None = None,
        prewarm_patch: bool = False,
        timeout_s: float = 24.0,
    ) -> dict[str, Any]:
        """Offload Earth-to-Sky zenith lookup off the ASGI event loop and prewarm in background."""
        loop = asyncio.get_running_loop()
        worker_fut = loop.run_in_executor(
            self._io_executor,
            functools.partial(
                self.get_earth_to_sky,
                lat=lat,
                lon=lon,
                iso_time=iso_time,
                fallback_seed=fallback_seed,
                prewarm_patch=False,
            ),
        )
        worker_fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        payload = await asyncio.wait_for(worker_fut, timeout=timeout_s)
        if prewarm_patch:
            earth = payload.get("earth", {})
            is_fallback = bool(earth.get("is_fallback_capital"))
            matched_capital = (
                True
                if is_fallback
                else find_matching_capital_city(float(earth["lat"]), float(earth["lon"]))
                is not None
            )
            if matched_capital:
                ra = float(payload["zenith_icrs"]["ra"])
                dec = float(payload["zenith_icrs"]["dec"])
                grid_key, _, _ = compute_grid_key(ra, dec, 5.0)
                cached_patch = self.cache.peek_manifest_memory(grid_key)
                if cached_patch is None:
                    cached_patch = await loop.run_in_executor(
                        self._io_executor, self.cache.get_manifest, grid_key
                    )
                needs_refresh = self.enable_archival_data and self._is_pending_archival_demo(
                    cached_patch
                )
                if cached_patch is None or needs_refresh:
                    self._schedule_background_patch_prewarm(ra=ra, dec=dec)
        return payload

    def _apply_cached_pipeline_sync(
        self,
        manifest: dict[str, Any],
        ra: float,
        dec: float,
        fov: float,
    ) -> dict[str, Any]:
        """Merge cached SIMBAD/Gaia/Horizons overlays into ``manifest`` synchronously."""
        pipeline_result = self.catalog_pipeline.resolve_cached_patch_objects(
            ra_deg=ra,
            dec_deg=dec,
            fov_arcmin=fov,
            wcs=manifest["wcs"],
            epochs_meta=manifest["epochs"],
        )
        return merge_pipeline_overlays(
            manifest,
            pipeline_result,
            prefer_pipeline_when_uncached=bool(pipeline_result.get("known_objects")),
        )

    def get_catalogs(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
    ) -> dict[str, Any]:
        """Synchronously resolve cached catalog overlays for a sky patch.

        Args:
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            fov: Field of view in arcminutes.

        Returns:
            Schema-validated ``CatalogsResponse`` dictionary.
        """
        grid_key, _, _ = compute_grid_key(ra, dec, fov)
        cached_catalogs = self.cache.get_catalogs(grid_key)
        if cached_catalogs is not None:
            validate_catalogs_response(cached_catalogs)
            return cached_catalogs

        manifest = build_sky_patch_manifest_payload(ra=ra, dec=dec, fov_arcmin=fov)
        manifest = self._apply_cached_pipeline_sync(manifest, ra=ra, dec=dec, fov=fov)
        response = {
            "query": manifest["query"],
            "wcs": manifest["wcs"],
            "catalog_overlays": manifest["catalog_overlays"],
        }
        validate_catalogs_response(response)
        return response

    async def _run_singleflight(
        self,
        key: str,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Coalesce concurrent async requests sharing ``key`` onto a single future."""
        loop = asyncio.get_running_loop()
        existing = self._inflight_tasks.get(key)
        if existing is not None and not existing.done() and existing.get_loop() is loop:
            _logger.info("Single-flight coalescing hit for %s", key)
            return await asyncio.shield(existing)

        fut: asyncio.Future[Any] = loop.create_future()
        fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self._inflight_tasks[key] = fut
        try:
            result = await coro_factory()
            if not fut.done():
                fut.set_result(result)
            return result
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            if self._inflight_tasks.get(key) is fut:
                self._inflight_tasks.pop(key, None)

    @staticmethod
    def _is_pending_archival_demo(manifest: dict[str, Any] | None) -> bool:
        """Return True if ``manifest`` is a placeholder simulated patch awaiting archival fetch."""
        if not manifest:
            return False
        summary = manifest.get("data_source_summary", {})
        note = str(summary.get("status_note", ""))
        return (
            note == "2 SPHEREx survey passes (6-month baseline)"
            or "archival lookup pending" in note.lower()
        )

    async def get_catalogs_async(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
        live_catalogs: bool = True,
    ) -> dict[str, Any]:
        """Asynchronously resolve catalog overlays with single-flight coalescing.

        Args:
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            fov: Field of view in arcminutes.
            live_catalogs: Whether live SIMBAD/Gaia queries are allowed on cache miss.

        Returns:
            Schema-validated ``CatalogsResponse`` dictionary.
        """
        loop = asyncio.get_running_loop()
        grid_key, _, _ = compute_grid_key(ra, dec, fov)
        cached_catalogs = self.cache.peek_catalogs_memory(grid_key)
        if cached_catalogs is None:
            cached_catalogs = await loop.run_in_executor(
                self._io_executor, self.cache.get_catalogs, grid_key
            )
        if cached_catalogs is not None:
            _logger.info("Catalogs cache hit for %s", grid_key)
            validate_catalogs_response(cached_catalogs)
            return cached_catalogs

        async def _build_catalogs() -> dict[str, Any]:
            manifest = build_sky_patch_manifest_payload(ra=ra, dec=dec, fov_arcmin=fov)
            pipeline_result = await self._run_singleflight(
                f"catalog_pipeline:{grid_key}:live{int(live_catalogs)}",
                lambda: self.catalog_pipeline.resolve_patch_objects(
                    ra_deg=ra,
                    dec_deg=dec,
                    fov_arcmin=fov,
                    wcs=manifest["wcs"],
                    epochs_meta=manifest["epochs"],
                    allow_live=live_catalogs,
                ),
            )
            merged = merge_pipeline_overlays(
                manifest,
                pipeline_result,
                prefer_pipeline_when_uncached=bool(pipeline_result.get("known_objects")),
            )
            response = {
                "query": merged["query"],
                "wcs": merged["wcs"],
                "catalog_overlays": merged["catalog_overlays"],
            }
            validate_catalogs_response(response)
            return response

        return await self._run_singleflight(
            f"catalogs:{grid_key}:live{int(live_catalogs)}",
            _build_catalogs,
        )

    def get_sky_patch(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Synchronously resolve, validate, and cache a sky-patch manifest and frames.

        Args:
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            fov: Field of view in arcminutes.
            client_ip: Optional client IP address for anonymous visitor deduplication.
            user_agent: Optional browser User-Agent string for visitor deduplication.

        Returns:
            Validated ``SkyPatchManifest`` dictionary.
        """
        grid_key, ra_q, dec_q = compute_grid_key(ra, dec, fov)
        visitor_token = self.cache.compute_visitor_token(client_ip, user_agent)
        cached_manifest = self.cache.get_manifest(grid_key)
        if cached_manifest is not None:
            is_legacy_demo = self._is_pending_archival_demo(cached_manifest)
            if not (self.enable_archival_data and is_legacy_demo):
                validate_sky_patch_manifest(cached_manifest)
                self.cache.record_patch_hit(grid_key, visitor_token)
                return cached_manifest

        stale_manifest = self.cache.get_manifest(grid_key, allow_stale=True)
        if stale_manifest is not None and self._is_pending_archival_demo(stale_manifest):
            stale_manifest = None

        manifest = build_sky_patch_manifest_payload(ra=ra, dec=dec, fov_arcmin=fov)
        manifest = self._apply_cached_pipeline_sync(manifest, ra=ra, dec=dec, fov=fov)
        slug = manifest["target_metadata"]["slug"]
        frames = generate_synthetic_epoch_frames(slug=slug, manifest=manifest)

        if self.enable_archival_data:
            try:
                manifest, frames = self.archival_resolver.resolve_archival_patch_sync(
                    manifest=manifest,
                    grid_key=grid_key,
                    ra=ra,
                    dec=dec,
                    fov=fov,
                    fallback_frames=frames,
                )
            except Exception as exc:
                if stale_manifest is not None:
                    _logger.warning(
                        "Archival refresh failed for %s (%s); serving stale cached patch",
                        grid_key,
                        exc,
                    )
                    validate_sky_patch_manifest(stale_manifest)
                    return stale_manifest
                raise
            if (
                stale_manifest is not None
                and manifest.get("data_source_summary", {}).get("provenance_tier") == "simulated"
                and stale_manifest.get("data_source_summary", {}).get("provenance_tier")
                != "simulated"
            ):
                validate_sky_patch_manifest(stale_manifest)
                return stale_manifest
            manifest = self._apply_cached_pipeline_sync(manifest, ra=ra, dec=dec, fov=fov)
            manifest, frames = apply_solar_system_ephemeris_to_frames(manifest, frames)

        validate_sky_patch_manifest(manifest)
        self.cache.put_sky_patch(
            grid_key=grid_key,
            ra_center=ra_q,
            dec_center=dec_q,
            fov_arcmin=fov,
            manifest=manifest,
            frames=frames,
            visitor_token=visitor_token,
        )
        return manifest

    async def get_sky_patch_async(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
        live_catalogs: bool = True,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously resolve, validate, and cache a sky-patch manifest and frames.

        Args:
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            fov: Field of view in arcminutes.
            live_catalogs: Whether live SIMBAD/Gaia queries are allowed on cache miss.
            client_ip: Optional client IP address for rate limiting & visitor deduplication.
            user_agent: Optional browser User-Agent string for visitor deduplication.

        Returns:
            Validated ``SkyPatchManifest`` dictionary.

        Raises:
            ColdMissRateLimitError: If ``client_ip`` exceeds the cold archival rate limit.
        """
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        grid_key, ra_q, dec_q = compute_grid_key(ra, dec, fov)
        visitor_token = self.cache.compute_visitor_token(client_ip, user_agent)
        cached_manifest = self.cache.peek_manifest_memory(grid_key)
        if cached_manifest is None:
            cached_manifest = await loop.run_in_executor(
                self._io_executor, self.cache.get_manifest, grid_key
            )
        if cached_manifest is not None:
            is_legacy_demo = self._is_pending_archival_demo(cached_manifest)
            if not (self.enable_archival_data and is_legacy_demo):
                _logger.info("Sky patch cache hit for %s", grid_key)
                validate_sky_patch_manifest(cached_manifest)
                await loop.run_in_executor(
                    self._io_executor,
                    self.cache.record_patch_hit,
                    grid_key,
                    visitor_token,
                )
                return cached_manifest

        stale_manifest: dict[str, Any] | None = None
        if cached_manifest is None:
            stale_manifest = self.cache.peek_manifest_memory(grid_key, allow_stale=True)
            if stale_manifest is None:
                stale_manifest = await loop.run_in_executor(
                    self._io_executor,
                    functools.partial(self.cache.get_manifest, grid_key, allow_stale=True),
                )
            if stale_manifest is not None and self._is_pending_archival_demo(stale_manifest):
                stale_manifest = None

        sf_key = f"patch:{grid_key}:arch{int(self.enable_archival_data)}:live{int(live_catalogs)}"
        existing_sf = self._inflight_tasks.get(sf_key)
        is_coalesced = (
            existing_sf is not None and not existing_sf.done() and existing_sf.get_loop() is loop
        )
        if client_ip and not is_coalesced:
            allowed, retry_after_s = self._cold_miss_limiter.allow(client_ip)
            if not allowed:
                if stale_manifest is not None:
                    validate_sky_patch_manifest(stale_manifest)
                    return stale_manifest
                raise ColdMissRateLimitError(
                    retry_after_s=retry_after_s,
                    limit_per_min=self._cold_miss_limiter.limit_per_min,
                )

        def _finalize_and_store_patch(
            base_manifest: dict[str, Any],
            resolved_manifest: dict[str, Any],
            frames: dict[str, bytes],
            pipeline_result: dict[str, Any],
        ) -> dict[str, Any]:
            epoch_ids_before = [e["epoch_id"] for e in base_manifest["epochs"]]
            epoch_ids_after = [e["epoch_id"] for e in resolved_manifest["epochs"]]
            if epoch_ids_after != epoch_ids_before:
                realigned = self.catalog_pipeline.resolve_cached_patch_objects(
                    ra_deg=ra,
                    dec_deg=dec,
                    fov_arcmin=fov,
                    wcs=resolved_manifest["wcs"],
                    epochs_meta=resolved_manifest["epochs"],
                )
                if realigned.get("known_objects"):
                    pipeline_result = realigned

            merged_manifest = merge_pipeline_overlays(
                resolved_manifest,
                pipeline_result,
                prefer_pipeline_when_uncached=bool(pipeline_result.get("known_objects")),
            )
            if self.enable_archival_data:
                merged_manifest, frames = apply_solar_system_ephemeris_to_frames(
                    merged_manifest, frames
                )

            self.cache.put_sky_patch(
                grid_key=grid_key,
                ra_center=ra_q,
                dec_center=dec_q,
                fov_arcmin=fov,
                manifest=merged_manifest,
                frames=frames,
                visitor_token=visitor_token,
            )
            validate_sky_patch_manifest(merged_manifest)
            return merged_manifest

        async def _compute_patch() -> dict[str, Any]:
            base_manifest = build_sky_patch_manifest_payload(ra=ra, dec=dec, fov_arcmin=fov)
            slug = base_manifest["target_metadata"]["slug"]

            catalog_coro = self._run_singleflight(
                f"catalog_pipeline:{grid_key}:live{int(live_catalogs)}",
                lambda: self.catalog_pipeline.resolve_patch_objects(
                    ra_deg=ra,
                    dec_deg=dec,
                    fov_arcmin=fov,
                    wcs=base_manifest["wcs"],
                    epochs_meta=base_manifest["epochs"],
                    allow_live=live_catalogs,
                ),
            )
            frames_fut = loop.run_in_executor(
                self._io_executor,
                functools.partial(
                    generate_synthetic_epoch_frames,
                    slug=slug,
                    manifest=base_manifest,
                ),
            )

            if self.enable_archival_data:
                archival_timeout_s = _env_float("ARCHIVAL_RESOLVE_BUDGET_S", 18.0) + 6.0

                async def _resolve_in_cpu_pool(
                    fallback_frames: dict[str, bytes],
                ) -> tuple[dict[str, Any], dict[str, bytes]]:
                    async with self._get_cpu_gate().slot(CPU_PRIORITY_SKY_PATCH):
                        cpu_fut = loop.run_in_executor(
                            self._cpu_executor,
                            functools.partial(
                                self.archival_resolver.resolve_archival_patch_sync,
                                manifest=base_manifest,
                                grid_key=grid_key,
                                ra=ra,
                                dec=dec,
                                fov=fov,
                                fallback_frames=fallback_frames,
                            ),
                        )
                        cpu_fut.add_done_callback(
                            lambda f: f.exception() if not f.cancelled() else None
                        )
                        return await cpu_fut

                async def _resolve_archival_bounded() -> tuple[dict[str, Any], dict[str, bytes]]:
                    fallback_frames = await frames_fut
                    try:
                        return await asyncio.wait_for(
                            _resolve_in_cpu_pool(fallback_frames),
                            timeout=archival_timeout_s,
                        )
                    except TimeoutError:
                        _logger.warning(
                            "Archival resolution for %s exceeded %.1fs; "
                            "serving labeled simulated baseline",
                            grid_key,
                            archival_timeout_s,
                        )
                        timed_out_manifest = dict(base_manifest)
                        timed_out_manifest["data_source_summary"] = {
                            "primary_mission": "Simulated",
                            "has_multi_epoch_spherex": False,
                            "uses_wise_fallback": False,
                            "baseline_days": float(
                                base_manifest["data_source_summary"].get("baseline_days", 182.7)
                            ),
                            "status_note": (
                                f"Archival retrieval timed out after {archival_timeout_s:.0f}s; "
                                "simulated baseline"
                            ),
                            "provenance_tier": "simulated",
                            "survey_name": "Simulated Multi-Band Baseline",
                        }
                        return timed_out_manifest, dict(fallback_frames)

                pipeline_result, (resolved_manifest, frames) = await asyncio.gather(
                    catalog_coro, _resolve_archival_bounded()
                )
                if (
                    stale_manifest is not None
                    and resolved_manifest.get("data_source_summary", {}).get("provenance_tier")
                    == "simulated"
                    and stale_manifest.get("data_source_summary", {}).get("provenance_tier")
                    != "simulated"
                ):
                    _logger.warning(
                        "Archival refresh degraded to simulated for %s; serving stale cached patch",
                        grid_key,
                    )
                    validate_sky_patch_manifest(stale_manifest)
                    return stale_manifest
            else:
                pipeline_result, frames = await asyncio.gather(catalog_coro, frames_fut)
                resolved_manifest = base_manifest

            merged_manifest = await loop.run_in_executor(
                self._io_executor,
                functools.partial(
                    _finalize_and_store_patch,
                    base_manifest,
                    resolved_manifest,
                    frames,
                    pipeline_result,
                ),
            )
            _logger.info(
                "Sky patch %s assembled (mission=%s, tier=%s, epochs=%d, known=%d, note=%s) "
                "in %.2fs",
                grid_key,
                merged_manifest["data_source_summary"].get("primary_mission"),
                merged_manifest["data_source_summary"].get("provenance_tier"),
                len(merged_manifest.get("epochs", [])),
                len(merged_manifest["catalog_overlays"].get("known_objects", [])),
                merged_manifest["data_source_summary"].get("status_note", ""),
                time.perf_counter() - t0,
            )
            return merged_manifest

        return await self._run_singleflight(sf_key, _compute_patch)

    def get_solar_system_in_fov(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
        mjd: float | None = None,
    ) -> dict[str, Any]:
        """Query cached Solar System ephemerides falling within the given field of view.

        Args:
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            fov: Field of view in arcminutes.
            mjd: Target Modified Julian Date (defaults to current UTC MJD).

        Returns:
            Dictionary containing matching ``bodies`` and ephemeris cache stats.
        """
        target_mjd = float(mjd) if mjd is not None else datetime_to_mjd()
        radius_deg = self.catalog_pipeline.fov_to_radius_deg(fov)
        matches = self.cache.query_ephemeris_in_fov(
            mjd=target_mjd,
            ra=ra,
            dec=dec,
            radius_deg=radius_deg,
        )
        return {
            "ra": round(float(ra), 5),
            "dec": round(float(dec), 5),
            "fov_arcmin": float(fov),
            "radius_deg": round(radius_deg, 4),
            "mjd": round(target_mjd, 6),
            "count": len(matches),
            "bodies": matches,
            "ephemeris_cache": self.cache.get_ephemeris_stats(),
        }

    async def get_solar_system_in_fov_async(
        self,
        ra: float,
        dec: float,
        fov: float = 5.0,
        mjd: float | None = None,
    ) -> dict[str, Any]:
        """Offload SQLite ephemeris cone lookup onto the dedicated I/O executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._io_executor,
            functools.partial(self.get_solar_system_in_fov, ra=ra, dec=dec, fov=fov, mjd=mjd),
        )

    async def lookup_most_object(self, obj_name: str) -> dict[str, Any]:
        """Query cached or live NASA/IPAC IRSA MOST intersections for a Solar System target.

        Args:
            obj_name: Target name or alias.

        Returns:
            Dictionary containing ``query_name``, ``status``, ``count``, and ``matches``.
        """
        cache_key = f"most:{obj_name.strip().lower()}"
        cached = self.cache.get_cached_catalog_region(cache_key, 0.0, 0.0, 1.0)
        if cached is not None:
            return {
                "query_name": cached.get("query_name", obj_name),
                "status": "cached",
                "count": len(cached.get("matches", [])),
                "matches": cached.get("matches", []),
            }
        result = await fetch_most_spherex_matches(
            obj_name=obj_name,
            client=self.catalog_pipeline.http_client,
        )
        if result["matches"]:
            self.cache.put_cached_catalog_region(
                cache_key,
                0.0,
                0.0,
                1.0,
                result,
                ttl_seconds=24 * 3600,
            )
        return {
            "query_name": result["query_name"],
            "status": result["status"],
            "count": len(result["matches"]),
            "matches": result["matches"],
        }

    def get_frame_bytes(
        self,
        grid_key: str,
        filename: str,
        precision: Literal["float32", "float16"] = "float32",
        accept_gzip: bool = False,
    ) -> tuple[bytes, dict[str, str]]:
        """Retrieve raw binary detector or flag bytes for a cached or synthetic frame.

        Args:
            grid_key: Quantized spatial grid identifier.
            filename: Binary frame filename (e.g. ``"pass_1.bin"``).
            precision: Output floating-point precision (``"float32"`` or ``"float16"``).
            accept_gzip: When True, serve pre-gzipped bytes with ``Content-Encoding: gzip``.

        Returns:
            Tuple of ``(body_bytes, response_headers)``.

        Raises:
            KeyError: If ``filename`` does not exist in cache or synthetic fallback.
        """
        is_flags = filename.endswith("_flags.bin")
        if accept_gzip and (precision == "float32" or is_flags):
            cached_gz = self.cache.get_frame_gzip(grid_key, filename, precision=precision)
            if cached_gz is not None:
                gz_bytes, dtype_header = cached_gz
                self.cache.record_frame_serve(dtype_header)
                return gz_bytes, {
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "X-Binary-Dtype": dtype_header,
                    "Content-Type": "application/octet-stream",
                    "Content-Encoding": "gzip",
                    "Vary": "Accept-Encoding",
                }

        raw: bytes | None = None
        if not is_flags:
            cached_gz32 = self.cache.get_frame_gzip(grid_key, filename, precision="float32")
            if cached_gz32 is not None:
                raw = gzip.decompress(cached_gz32[0])
        if raw is None:
            raw = self.cache.get_frame(grid_key, filename)
        if raw is None:
            cached_manifest = self.cache.get_manifest(grid_key)
            if cached_manifest is not None:
                frames = generate_synthetic_epoch_frames(
                    slug=cached_manifest["target_metadata"]["slug"],
                    manifest=cached_manifest,
                )
            else:
                frames = generate_synthetic_epoch_frames("wise_0855")
            if filename not in frames:
                raise KeyError(f"Frame not found: {grid_key}/{filename}")
            raw = frames[filename]

        if is_flags:
            dtype_header = "uint8"
            body = raw
        elif precision == "float16":
            dtype_header = "float16"
            body = convert_float32_bytes_to_float16_bytes(raw)
        else:
            dtype_header = "float32"
            body = raw

        self.cache.record_frame_serve(dtype_header)
        headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Binary-Dtype": dtype_header,
            "Content-Type": "application/octet-stream",
        }
        if accept_gzip:
            compress_lvl = 1 if dtype_header == "float16" else 3
            body = gzip.compress(body, compresslevel=compress_lvl)
            headers["Content-Encoding"] = "gzip"
            headers["Vary"] = "Accept-Encoding"
        return body, headers

    async def get_frame_bytes_async(
        self,
        grid_key: str,
        filename: str,
        precision: Literal["float32", "float16"] = "float32",
        accept_gzip: bool = False,
    ) -> tuple[bytes, dict[str, str]]:
        """Serve RAM LRU frame hits synchronously; offload SQLite/conversion to ``_io_executor``."""
        if accept_gzip and (precision == "float32" or filename.endswith("_flags.bin")):
            peeked = self.cache.peek_frame_gzip_memory(grid_key, filename, precision=precision)
            if peeked is not None:
                gz_bytes, dtype_header = peeked
                self.cache.record_frame_serve(dtype_header)
                return gz_bytes, {
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "X-Binary-Dtype": dtype_header,
                    "Content-Type": "application/octet-stream",
                    "Content-Encoding": "gzip",
                    "Vary": "Accept-Encoding",
                }
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._io_executor,
            functools.partial(
                self.get_frame_bytes,
                grid_key=grid_key,
                filename=filename,
                precision=precision,
                accept_gzip=accept_gzip,
            ),
        )

    def get_forced_spectrum(
        self,
        ra: float,
        dec: float,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Ingest SPHEREx cutouts and measure a forced 102-band spectrum at ``(ra, dec)``.

        Args:
            ra: Target Right Ascension in degrees.
            dec: Target Declination in degrees.
            label: Optional human-readable target label; defaults to
                ``"Selected sky position"`` when omitted or blank.

        Returns:
            Validated 102-channel ``SpectrumPayload`` dictionary.
        """
        clean_label = label.strip() if label and label.strip() else None
        resolved_label = clean_label or "Selected sky position"
        spectrum_key = f"spec_forced_ra{round(ra, 4):.4f}_dec{round(dec, 4):.4f}"
        cached_spec = self.cache.get_spectrum(spectrum_key)
        if cached_spec is not None:
            cached_spec["source_label"] = resolved_label
            validate_spectrum_payload(cached_spec)
            return cached_spec

        stale_spec = self.cache.get_spectrum(spectrum_key, allow_stale=True)
        try:
            try:
                cutouts = self.spherex_cutout_fetcher(
                    ra=ra,
                    dec=dec,
                    fov_arcmin=5.0,
                    allow_synthetic_fallback=True,
                    max_cutouts=DEFAULT_MAX_SPECTRUM_CUTOUTS,
                    budget_seconds=13.0,
                )
            except TypeError:
                cutouts = self.spherex_cutout_fetcher(
                    ra=ra,
                    dec=dec,
                    fov_arcmin=5.0,
                    allow_synthetic_fallback=True,
                )
            payload = self.spectrum_extractor(
                ra=ra,
                dec=dec,
                cutouts=cutouts,
                source_label=resolved_label,
            )
        except Exception as exc:
            if stale_spec is not None:
                _logger.warning(
                    "Spectrum refresh failed for %s (%s); serving stale cached spectrum",
                    spectrum_key,
                    exc,
                )
                stale_spec["source_label"] = resolved_label
                validate_spectrum_payload(stale_spec)
                return stale_spec
            raise
        self.cache.put_spectrum(spectrum_key=spectrum_key, ra=ra, dec=dec, payload=payload)
        return payload

    async def get_forced_spectrum_async(
        self,
        ra: float,
        dec: float,
        label: str | None = None,
        timeout_s: float = 20.0,
        client_ip: str | None = None,
    ) -> dict[str, Any]:
        """Fetch or compute a 102-band forced spectrum using separated I/O and CPU pools."""
        clean_label = label.strip() if label and label.strip() else None
        resolved_label = clean_label or "Selected sky position"
        spectrum_key = f"spec_forced_ra{round(ra, 4):.4f}_dec{round(dec, 4):.4f}"
        loop = asyncio.get_running_loop()
        cached_spec = self.cache.peek_spectrum_memory(spectrum_key)
        if cached_spec is None:
            cached_spec = await loop.run_in_executor(
                self._io_executor, self.cache.get_spectrum, spectrum_key
            )
        if cached_spec is not None:
            cached_spec["source_label"] = resolved_label
            validate_spectrum_payload(cached_spec)
            return cached_spec

        sf_key = f"spectrum:{spectrum_key}"
        existing_sf = self._inflight_tasks.get(sf_key)
        is_coalesced = (
            existing_sf is not None and not existing_sf.done() and existing_sf.get_loop() is loop
        )
        if client_ip and not is_coalesced:
            allowed, retry_after_s = self._cold_miss_limiter.allow(client_ip)
            if not allowed:
                stale_spec = self.cache.peek_spectrum_memory(spectrum_key, allow_stale=True)
                if stale_spec is None:
                    stale_spec = await loop.run_in_executor(
                        self._io_executor,
                        functools.partial(self.cache.get_spectrum, spectrum_key, allow_stale=True),
                    )
                if stale_spec is not None:
                    stale_spec["source_label"] = resolved_label
                    validate_spectrum_payload(stale_spec)
                    return stale_spec
                raise ColdMissRateLimitError(
                    retry_after_s=retry_after_s,
                    limit_per_min=self._cold_miss_limiter.limit_per_min,
                )

        async def _compute() -> dict[str, Any]:
            async def _run_in_cpu_pool() -> dict[str, Any]:
                async with self._get_cpu_gate().slot(CPU_PRIORITY_SPECTRUM):
                    cpu_fut = loop.run_in_executor(
                        self._cpu_executor,
                        functools.partial(
                            self.get_forced_spectrum,
                            ra=ra,
                            dec=dec,
                            label=resolved_label,
                        ),
                    )
                    cpu_fut.add_done_callback(
                        lambda f: f.exception() if not f.cancelled() else None
                    )
                    return await cpu_fut

            return await asyncio.wait_for(_run_in_cpu_pool(), timeout=timeout_s)

        payload = await self._run_singleflight(sf_key, _compute)
        if payload.get("source_label") != resolved_label:
            payload = dict(payload)
            payload["source_label"] = resolved_label
        return payload

    def get_spectrum(
        self,
        ra: float,
        dec: float,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Return a cached or showcase-tailored 102-band spectrum at ``(ra, dec)``.

        Args:
            ra: Target Right Ascension in degrees.
            dec: Target Declination in degrees.
            label: Optional target label.

        Returns:
            Validated 102-channel ``SpectrumPayload`` dictionary.
        """
        spectrum_key = f"spec_ra{round(ra, 4):.4f}_dec{round(dec, 4):.4f}"
        cached_spec = self.cache.get_spectrum(spectrum_key)
        if cached_spec is not None:
            validate_spectrum_payload(cached_spec)
            return cached_spec

        resolved_label = label or "WISE 0855"
        resolved_slug: str | None = None
        for cand_slug, meta in SHOWCASE_TARGETS.items():
            if math.hypot(ra - meta["ra"], dec - meta["dec"]) < 0.25:
                resolved_label = meta["name"]
                resolved_slug = cand_slug
                break
        payload = build_spectrum_payload(
            ra=ra,
            dec=dec,
            source_label=resolved_label,
            slug=resolved_slug,
        )
        self.cache.put_spectrum(spectrum_key=spectrum_key, ra=ra, dec=dec, payload=payload)
        return payload
