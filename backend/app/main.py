"""FastAPI application factory and HTTP route definitions for Star Hunters."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response as StarletteResponse
from starlette.types import Scope

from app.services.backend_service import ColdMissRateLimitError, StarHuntersBackendService
from app.services.logging_config import configure_backend_logging
from app.services.spectrum_extractor import SpectrumExtractionError

_HTTP_CLIENT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_HTTP_CLIENT_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=16)


_SINGLE_IP_PROXY_HEADERS = (
    "fly-client-ip",
    "cf-connecting-ip",
    "true-client-ip",
    "x-real-ip",
)


def _normalize_ip(raw: str) -> str | None:
    """Parse and normalize a candidate IPv4/IPv6 string (stripping quotes/ports)."""
    candidate = raw.strip().strip('"').strip("'")
    if not candidate:
        return None
    if candidate.startswith("["):
        end_bracket = candidate.find("]")
        if end_bracket == -1:
            return None
        candidate = candidate[1:end_bracket]
    elif candidate.count(":") == 1 and "." in candidate:
        # IPv4:port
        candidate = candidate.rsplit(":", 1)[0]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def _extract_forwarded_for(header_value: str) -> str | None:
    """Extract the leftmost valid client IP from an RFC 7239 Forwarded header."""
    for entry in header_value.split(","):
        for param in entry.split(";"):
            name, sep, value = param.strip().partition("=")
            if sep and name.strip().lower() == "for":
                ip = _normalize_ip(value)
                if ip is not None:
                    return ip
    return None


def _extract_client_ip(request: Request) -> str:
    """Resolve originating client IP across standard reverse-proxy and platform headers."""
    for header_name in _SINGLE_IP_PROXY_HEADERS:
        raw_value = request.headers.get(header_name)
        if raw_value:
            ip = _normalize_ip(raw_value)
            if ip is not None:
                return ip

    forwarded = request.headers.get("forwarded")
    if forwarded:
        ip = _extract_forwarded_for(forwarded)
        if ip is not None:
            return ip

    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        for part in x_forwarded_for.split(","):
            ip = _normalize_ip(part)
            if ip is not None:
                return ip

    if request.client and request.client.host:
        host = request.client.host.strip()
        if host:
            return _normalize_ip(host) or host
    return "unknown"


class CachedStaticFiles(StaticFiles):
    """StaticFiles mount setting immutable headers on hashed assets and no-cache on HTML."""

    async def get_response(self, path: str, scope: Scope) -> StarletteResponse:
        """Return static file response with long-lived cache headers for Vite assets."""
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            normalized = path.lstrip("./")
            if normalized.startswith("assets/") or "/assets/" in f"/{normalized}":
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            elif (
                normalized in {"", "index.html"}
                or normalized.endswith(".html")
                or "text/html" in response.headers.get("content-type", "")
            ):
                response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(service: StarHuntersBackendService | None = None) -> FastAPI:
    """Create and configure the Star Hunters FastAPI application.

    Args:
        service: Optional pre-configured :class:`StarHuntersBackendService`
            instance (useful for dependency-injected testing). When omitted, a
            default service backed by :class:`SpatialSqliteCache` is created.

    Returns:
        Configured :class:`fastapi.FastAPI` instance with API routes, middleware,
        lifespan cleanup, and optional built SPA static serving.
    """
    active_service = service or StarHuntersBackendService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Manage unified logging, pooled HTTP client, and SQLite pool across ASGI lifespan."""
        configure_backend_logging()
        managed_http_client: httpx.AsyncClient | None = None
        if active_service.catalog_pipeline.http_client is None:
            managed_http_client = httpx.AsyncClient(
                timeout=_HTTP_CLIENT_TIMEOUT,
                limits=_HTTP_CLIENT_LIMITS,
                follow_redirects=True,
            )
            active_service.catalog_pipeline.http_client = managed_http_client
        try:
            yield
        finally:
            active_service.cancel_background_tasks()
            try:
                if managed_http_client is not None:
                    if active_service.catalog_pipeline.http_client is managed_http_client:
                        active_service.catalog_pipeline.http_client = None
                    await managed_http_client.aclose()
            finally:
                active_service.cache.close()

    fastapi_app = FastAPI(
        title="Star Hunters API (Planet X & SPHEREx Blinker)",
        version="0.1.0",
        description="Multi-epoch SPHEREx + WISE/Gaia/MOST sky blinker & 102-band spectral API.",
        lifespan=lifespan,
    )
    fastapi_app.state.service = active_service
    fastapi_app.add_middleware(GZipMiddleware, minimum_size=512)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "Authorization", "X-Cache-Delta"],
        expose_headers=["X-Binary-Dtype"],
    )

    @fastapi_app.get("/health")
    async def health_check() -> dict[str, str]:
        """Return backend liveness status."""
        return active_service.health()

    @fastapi_app.get("/api/v1/earth-to-sky")
    @fastapi_app.get("/api/earth-to-sky")
    async def earth_to_sky(
        lat: float | None = Query(None),
        lon: float | None = Query(None),
        iso_time: str | None = Query(None),
        time: str | None = Query(None),
    ) -> dict[str, Any]:
        """Compute overhead ICRS zenith coordinates for an Earth location and UTC time.

        Args:
            lat: Observer latitude in degrees (falls back to random capital if omitted).
            lon: Observer longitude in degrees.
            iso_time: Primary ISO-8601 UTC observation timestamp.
            time: Optional alias fallback when ``iso_time`` is omitted.

        Returns:
            Validated ``EarthToSkyResponse`` dictionary.
        """
        resolved_time = iso_time or time
        try:
            return await active_service.get_earth_to_sky_async(
                lat=lat, lon=lon, iso_time=resolved_time
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fastapi_app.get("/api/v1/catalogs")
    @fastapi_app.get("/api/catalogs")
    async def get_catalogs(
        request: Request,
        response: Response,
        ra: float = Query(..., ge=0.0, le=360.0),
        dec: float = Query(..., ge=-90.0, le=90.0),
        fov: float = Query(5.0, gt=0.0, le=60.0),
        live_catalogs: bool = Query(True),
    ) -> dict[str, Any]:
        """Return multi-catalog overlays (SIMBAD, Gaia DR3, Solar System) for a sky patch."""
        active_service.cache.record_client_cache_delta(request.headers.get("x-cache-delta"))
        response.headers["Cache-Control"] = "public, max-age=3600"
        return await active_service.get_catalogs_async(
            ra=ra,
            dec=dec,
            fov=fov,
            live_catalogs=live_catalogs,
        )

    @fastapi_app.get("/api/v1/catalogs/search")
    @fastapi_app.get("/api/catalogs/search")
    async def search_catalog_objects(
        response: Response,
        q: str = Query("", description="Fuzzy search query for celestial object names/IDs"),
        limit: int = Query(30, ge=1, le=200),
    ) -> dict[str, Any]:
        """Fuzzy-search known celestial objects and refresh live Horizons ephemerides."""
        response.headers["Cache-Control"] = "public, max-age=3600"
        results = await active_service.search_known_objects_async(query=q, limit=limit)
        return {
            "query": q,
            "count": len(results),
            "results": results,
        }

    @fastapi_app.get("/api/v1/catalogs/object-epochs")
    @fastapi_app.get("/api/catalogs/object-epochs")
    async def get_catalog_object_epochs(
        response: Response,
        object_id: str = Query(""),
        label: str = Query(..., min_length=1),
        object_type: str | None = Query(None),
        object_class: str | None = Query(None),
        target_slug: str | None = Query(None),
        ra: float | None = Query(None, ge=0.0, le=360.0),
        dec: float | None = Query(None, ge=-90.0, le=90.0),
    ) -> dict[str, Any]:
        """Return constrained epochs (Current Time + SPHEREx passes only) for a target."""
        response.headers["Cache-Control"] = "public, max-age=1800"
        return await active_service.get_object_observation_epochs_async(
            object_id=object_id,
            label=label,
            object_type=object_type,
            object_class=object_class,
            target_slug=target_slug,
            ra=ra,
            dec=dec,
        )

    @fastapi_app.get("/api/v1/solar-system/in-fov")
    @fastapi_app.get("/api/solar-system/in-fov")
    async def solar_system_in_fov(
        ra: float = Query(..., ge=0.0, le=360.0),
        dec: float = Query(..., ge=-90.0, le=90.0),
        fov: float = Query(5.0, gt=0.0, le=60.0),
        mjd: float | None = Query(None),
    ) -> dict[str, Any]:
        """Query cached Solar System ephemerides falling within the requested FOV."""
        return await active_service.get_solar_system_in_fov_async(ra=ra, dec=dec, fov=fov, mjd=mjd)

    @fastapi_app.get("/api/v1/solar-system/most")
    @fastapi_app.get("/api/solar-system/most")
    async def lookup_most_solar_system(
        response: Response,
        obj_name: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        """Query NASA/IPAC IRSA MOST for SPHEREx intersections with a Solar System body."""
        response.headers["Cache-Control"] = "public, max-age=3600"
        return await active_service.lookup_most_object(obj_name=obj_name)

    @fastapi_app.get("/api/v1/sky-patch")
    @fastapi_app.get("/api/sky-patch")
    async def get_sky_patch(
        request: Request,
        response: Response,
        ra: float = Query(..., ge=0.0, le=360.0),
        dec: float = Query(..., ge=-90.0, le=90.0),
        fov: float = Query(5.0, gt=0.0, le=60.0),
        live_catalogs: bool = Query(True),
    ) -> dict[str, Any]:
        """Assemble or retrieve a cached multi-epoch ``SkyPatchManifest``."""
        active_service.cache.record_client_cache_delta(request.headers.get("x-cache-delta"))
        response.headers["Cache-Control"] = "public, max-age=3600"
        try:
            return await active_service.get_sky_patch_async(
                ra=ra,
                dec=dec,
                fov=fov,
                live_catalogs=live_catalogs,
                client_ip=_extract_client_ip(request),
                user_agent=request.headers.get("user-agent"),
            )
        except ColdMissRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(max(1, math.ceil(exc.retry_after_s)))},
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Sky patch assembly unavailable: {exc}"
            ) from exc

    @fastapi_app.get("/api/v1/frames/{grid_key}/{filename}")
    @fastapi_app.get("/api/frames/{grid_key}/{filename}")
    async def get_binary_frame(
        grid_key: str,
        filename: str,
        precision: Literal["float32", "float16"] = Query("float32"),
        accept_encoding: str | None = Header(None),
    ) -> Response:
        """Stream a packed binary detector cube or uint8 hygiene mask."""
        accept_gzip = "gzip" in (accept_encoding or "").lower()
        try:
            body, headers = await active_service.get_frame_bytes_async(
                grid_key=grid_key,
                filename=filename,
                precision=precision,
                accept_gzip=accept_gzip,
            )
            return Response(
                content=body,
                media_type="application/octet-stream",
                headers=headers,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @fastapi_app.get("/api/v1/spectrum")
    @fastapi_app.get("/api/spectrum")
    async def get_spectrum(
        request: Request,
        response: Response,
        ra: float = Query(..., ge=0.0, le=360.0),
        dec: float = Query(..., ge=-90.0, le=90.0),
        label: str | None = Query(None),
    ) -> dict[str, Any]:
        """Extract or retrieve a 102-channel SPHEREx spectrum at ``(ra, dec)``.

        Args:
            request: Incoming FastAPI request used to resolve client IP.
            response: FastAPI response object for cache-control headers.
            ra: Target Right Ascension in degrees.
            dec: Target Declination in degrees.
            label: Optional target label forwarded as ``source_label``.

        Returns:
            Validated 102-channel ``SpectrumPayload`` dictionary.
        """
        response.headers["Cache-Control"] = "public, max-age=3600"
        try:
            return await active_service.get_forced_spectrum_async(
                ra=ra,
                dec=dec,
                label=label,
                client_ip=_extract_client_ip(request),
            )
        except ColdMissRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(max(1, math.ceil(exc.retry_after_s)))},
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="SPHEREx spectrum extraction timed out; upstream archive unavailable.",
            ) from exc
        except SpectrumExtractionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"SPHEREx spectrum unavailable: {exc}"
            ) from exc

    @fastapi_app.get("/api/v1/cache-stats")
    @fastapi_app.get("/api/cache-stats")
    async def get_cache_stats(
        response: Response,
        authorization: str | None = Header(None),
    ) -> dict[str, Any]:
        """Return aggregate cache telemetry ratios, occupancy, and top cached patches."""
        expected_token = os.environ.get("CACHE_STATS_TOKEN", "").strip()
        if expected_token:
            expected_header = f"Bearer {expected_token}"
            if not authorization or not hmac.compare_digest(authorization, expected_header):
                raise HTTPException(status_code=403, detail="Invalid or missing CACHE_STATS_TOKEN")
        response.headers["Cache-Control"] = "no-store"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            active_service._io_executor, active_service.cache.get_cache_stats
        )

    @fastapi_app.post("/api/v1/admin/clear-cache")
    async def clear_cache(
        x_admin_token: str | None = Header(None),
    ) -> dict[str, str]:
        """Clear all SQLite spatial cache tables when authorized via ``X-Admin-Token``."""
        if x_admin_token != active_service.admin_token:
            raise HTTPException(status_code=403, detail="Invalid X-Admin-Token")
        active_service.cancel_background_tasks()
        active_service.cache.clear()
        return {"status": "cleared"}

    frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        fastapi_app.mount(
            "/",
            CachedStaticFiles(directory=str(frontend_dist), html=True),
            name="frontend_spa",
        )

    return fastapi_app


app = create_app()
