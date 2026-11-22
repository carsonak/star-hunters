"""Shared async HTTP load helpers and endpoint summary aggregation."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict
from typing import Any

from lib.host import percentiles


async def timed_get(
    client: Any,
    url: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    endpoint: str,
    *,
    headers: dict[str, str] | None = None,
    track_cached: bool = False,
) -> Any | None:
    """Issue a GET request and append latency, status, and byte-size telemetry."""
    import httpx

    t0 = time.perf_counter()
    try:
        resp = await client.get(url, params=params, headers=headers)
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


async def fetch_patch_and_frames(
    client: Any,
    ra: float,
    dec: float,
    records: list[dict[str, Any]],
    *,
    catalogs_endpoint: str,
    sky_patch_endpoint: str,
    frame_endpoint: str,
    spectrum_endpoint: str,
    fetch_spectrum: bool = False,
    spectrum_label: str | None = None,
    precision: str = "float16",
    frame_headers: dict[str, str] | None = None,
    track_cached: bool = False,
) -> int:
    """Fetch Stage A catalogs, Stage B sky-patch manifest, binary frames, and optional spectrum."""
    await timed_get(
        client,
        "/api/v1/catalogs",
        {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
        records,
        catalogs_endpoint,
    )
    patch_resp = await timed_get(
        client,
        "/api/v1/sky-patch",
        {"ra": ra, "dec": dec, "fov": 5.0, "live_catalogs": "false"},
        records,
        sky_patch_endpoint,
        track_cached=track_cached,
    )
    requests_made = 2

    if patch_resp is not None and patch_resp.status_code == 200:
        manifest = patch_resp.json()
        frame_urls: list[tuple[str, dict[str, Any]]] = []
        for ep in manifest.get("epochs", []):
            det_url = ep.get("detectors_data_url")
            flg_url = ep.get("flags_mask_url")
            if det_url:
                frame_urls.append((det_url, {"precision": precision}))
            if flg_url:
                frame_urls.append((flg_url, {}))
        if frame_urls:
            await asyncio.gather(
                *(
                    timed_get(
                        client,
                        url,
                        params,
                        records,
                        frame_endpoint,
                        headers=frame_headers,
                    )
                    for url, params in frame_urls
                )
            )
            requests_made += len(frame_urls)

    if fetch_spectrum:
        await timed_get(
            client,
            "/api/v1/spectrum",
            {
                "ra": ra,
                "dec": dec,
                "label": spectrum_label or f"Target {ra:.2f},{dec:.2f}",
            },
            records,
            spectrum_endpoint,
        )
        requests_made += 1

    return requests_made


def summarize_endpoints(
    records: list[dict[str, Any]],
    *,
    include_extended: bool = False,
) -> dict[str, Any]:
    """Aggregate per-endpoint counts, errors, and latency percentiles from raw request records."""
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_ep[rec["endpoint"]].append(rec)

    summary: dict[str, Any] = {}
    for ep, rows in sorted(by_ep.items()):
        lats = [r["ms"] for r in rows]
        p50, p95, p99 = percentiles(lats, (50.0, 95.0, 99.0))
        errors = sum(1 for r in rows if r["status"] >= 400)
        entry: dict[str, Any] = {
            "count": len(rows),
            "errors": errors,
            "mean_ms": round(statistics.mean(lats), 2) if lats else 0.0,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "max_ms": round(max(lats), 2) if lats else 0.0,
        }
        if include_extended:
            wire_b = sum(r["wire_bytes"] for r in rows)
            raw_b = sum(r["raw_bytes"] for r in rows)
            entry.update(
                {
                    "timeouts": sum(1 for r in rows if r["status"] == 599),
                    "over_45s": sum(1 for r in rows if r["ms"] >= 45_000.0),
                    "wire_mb": round(wire_b / (1024 * 1024), 3),
                    "raw_mb": round(raw_b / (1024 * 1024), 3),
                }
            )
        summary[ep] = entry
    return summary
