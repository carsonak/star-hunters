"""Quantized SQLite spatial cache for preprocessed sky patches, frames & spectra."""

from __future__ import annotations

import copy
import gzip
import hashlib
import hmac
import json
import math
import os
import sqlite3
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from app.models.contracts import BINARY_DETECTOR_FLOAT32_BYTES
from app.utils.astronomy import (
    compute_ephemeris_fov_bounds,
    filter_ephemeris_candidates_in_fov,
    interpolate_ephemeris_points,
    mjd_to_date_bucket,
)

DEFAULT_GRID_STEP_DEG = 0.05  # 3 arcminutes
DEFAULT_MAX_CACHE_BYTES = 750 * 1024 * 1024  # 750 MB
DEFAULT_MAX_MEMORY_CACHE_BYTES = 128 * 1024 * 1024  # 128 MB
DEFAULT_CATALOG_TTL_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_SKY_PATCH_TTL_SECONDS = 7 * 24 * 3600  # 7 days (weekly SPHEREx IRSA releases)
DEFAULT_POPULARITY_HALF_LIFE_SECONDS = 7 * 24 * 3600.0  # 7-day exponential decay half-life
POPULARITY_EPOCH_T0 = 1700000000.0  # Fixed anchor epoch so rank comparisons cancel current time t
MAX_RECENT_VISITOR_HASHES = 128  # 128 x 4-byte hashes = 512 bytes max per patch
LAST_ACCESS_UPDATE_INTERVAL_SECONDS = 300.0  # 5 minutes
SCHEMA_VERSION = 2
_MIGRATIONS_DIR = Path(__file__).with_name("migrations")

TELEMETRY_METRICS = (
    "client_local_patch_hits",
    "client_local_catalog_hits",
    "common_multi_user_hits",
    "same_user_backend_repeats",
    "unique_cold_misses",
    "float32_frame_serves",
    "float16_frame_serves",
)


def _resolve_max_cache_bytes(max_cache_bytes: int | None) -> int:
    """Resolve SQLite byte budget from argument or ``SQLITE_MAX_CACHE_MB``."""
    if max_cache_bytes is not None:
        return max(0, int(max_cache_bytes))
    raw = os.environ.get("SQLITE_MAX_CACHE_MB", "").strip()
    if not raw:
        return DEFAULT_MAX_CACHE_BYTES
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:
        return DEFAULT_MAX_CACHE_BYTES


def _resolve_max_memory_bytes(max_memory_bytes: int | None) -> int:
    """Resolve the in-process LRU byte budget from argument or ``IN_MEMORY_CACHE_MB``."""
    if max_memory_bytes is not None:
        return max(0, int(max_memory_bytes))
    raw = os.environ.get("IN_MEMORY_CACHE_MB", "").strip()
    if not raw:
        return DEFAULT_MAX_MEMORY_CACHE_BYTES
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:
        return DEFAULT_MAX_MEMORY_CACHE_BYTES


def _resolve_patch_ttl_seconds(patch_ttl_seconds: float | None) -> float:
    """Resolve sky-patch & spectrum TTL from argument or ``SKY_PATCH_TTL_SECONDS``."""
    if patch_ttl_seconds is not None:
        return max(0.0, float(patch_ttl_seconds))
    raw = os.environ.get("SKY_PATCH_TTL_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_SKY_PATCH_TTL_SECONDS)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_SKY_PATCH_TTL_SECONDS)


def _resolve_popularity_half_life_seconds(half_life_seconds: float | None) -> float:
    """Resolve exponential popularity decay half-life from argument or env var."""
    if half_life_seconds is not None:
        return max(60.0, float(half_life_seconds))
    raw = os.environ.get("POPULARITY_HALF_LIFE_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_POPULARITY_HALF_LIFE_SECONDS)
    try:
        return max(60.0, float(raw))
    except ValueError:
        return float(DEFAULT_POPULARITY_HALF_LIFE_SECONDS)


def _visitor_in_ring(ring: bytes, token: bytes) -> bool:
    """Return True if 4-byte ``token`` is present at any 4-byte aligned slot in ``ring``."""
    if len(token) != 4 or not ring:
        return False
    return any(ring[i : i + 4] == token for i in range(0, len(ring) - (len(ring) % 4), 4))


def _append_visitor_ring(ring: bytes, token: bytes) -> bytes:
    """Append 4-byte ``token`` to ``ring``, keeping at most ``MAX_RECENT_VISITOR_HASHES`` slots."""
    if len(token) != 4:
        return ring
    max_bytes = MAX_RECENT_VISITOR_HASHES * 4
    updated = ring + token
    return updated[-max_bytes:] if len(updated) > max_bytes else updated


class _MemoryLruCache:
    """Thread-safe, byte-budgeted in-process LRU cache storing immutable ``bytes``."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._tag_to_keys: dict[str, set[str]] = {}
        self._current_bytes = 0

    @staticmethod
    def _grid_tag_for_key(key: str) -> str | None:
        """Extract owning ``grid_key`` tag for O(1) multi-key patch invalidation."""
        if key.startswith(("manifest:", "catalogs:", "frame_raw:", "frame_gz:")):
            parts = key.split(":", 2)
            if len(parts) >= 2 and parts[1]:
                return parts[1]
        return None

    def _index_key(self, key: str) -> None:
        tag = self._grid_tag_for_key(key)
        if tag is not None:
            self._tag_to_keys.setdefault(tag, set()).add(key)

    def _unindex_key(self, key: str) -> None:
        tag = self._grid_tag_for_key(key)
        if tag is not None:
            keys = self._tag_to_keys.get(tag)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._tag_to_keys.pop(tag, None)

    @property
    def current_bytes(self) -> int:
        """Total bytes currently held by cached entries."""
        with self._lock:
            return self._current_bytes

    @property
    def entry_count(self) -> int:
        """Number of entries currently held in the in-process LRU cache."""
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> bytes | None:
        """Return cached bytes and mark ``key`` as most recently used, or ``None``."""
        if self.max_bytes <= 0:
            return None
        with self._lock:
            val = self._entries.get(key)
            if val is None:
                return None
            self._entries.move_to_end(key)
            return val

    def put(self, key: str, value: bytes) -> None:
        """Insert ``value`` under ``key``, evicting oldest entries to stay under ``max_bytes``."""
        if self.max_bytes <= 0:
            return
        size = len(value)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._current_bytes -= len(old)
                self._unindex_key(key)
            if size > self.max_bytes:
                return
            while self._entries and self._current_bytes + size > self.max_bytes:
                evicted_key, evicted = self._entries.popitem(last=False)
                self._current_bytes -= len(evicted)
                self._unindex_key(evicted_key)
            self._entries[key] = value
            self._current_bytes += size
            self._index_key(key)

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._current_bytes -= len(old)
                self._unindex_key(key)

    def delete_tag(self, tag: str) -> None:
        """Remove all cached entries associated with ``tag`` in O(k) time."""
        with self._lock:
            keys = self._tag_to_keys.pop(tag, None)
            if not keys:
                return
            for key in keys:
                old = self._entries.pop(key, None)
                if old is not None:
                    self._current_bytes -= len(old)

    def delete_prefix(self, prefix: str) -> None:
        """Remove all entries whose keys begin with ``prefix``."""
        with self._lock:
            keys = [key for key in self._entries if key.startswith(prefix)]
            for key in keys:
                old = self._entries.pop(key)
                self._current_bytes -= len(old)
                self._unindex_key(key)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._entries.clear()
            self._tag_to_keys.clear()
            self._current_bytes = 0


def quantize_bucket(
    value: float, step: float = DEFAULT_GRID_STEP_DEG, wrap_360: bool = False
) -> float:
    """Quantize a celestial coordinate to a spatial bucket step in degrees.

    Args:
        value: Input coordinate in degrees.
        step: Quantization grid step in degrees.
        wrap_360: If True, wrap coordinate modulo 360 (Right Ascension);
            otherwise clamp to ``[-90, 90]`` (Declination).

    Returns:
        Quantized coordinate value in degrees.
    """
    val = float(value)
    if wrap_360:
        val = val % 360.0
    else:
        val = max(-90.0, min(90.0, val))
    q = round(round(val / step) * step, 4)
    return q % 360.0 if wrap_360 else max(-90.0, min(90.0, q))


def compute_grid_key(
    ra: float,
    dec: float,
    fov_arcmin: float = 5.0,
    grid_step_deg: float = DEFAULT_GRID_STEP_DEG,
) -> tuple[str, float, float]:
    """Quantize ``(ra, dec)`` to a 0.05-deg spatial grid and build its cache key.

    Args:
        ra: Right Ascension in degrees.
        dec: Declination in degrees.
        fov_arcmin: Field of view in arcminutes.
        grid_step_deg: Quantization step in degrees.

    Returns:
        Tuple of ``(grid_key, ra_quantized, dec_quantized)``.
    """
    normalized_ra = ra % 360.0
    clamped_dec = max(-90.0, min(90.0, dec))
    ra_q = round(round(normalized_ra / grid_step_deg) * grid_step_deg, 2) % 360.0
    dec_q = round(round(clamped_dec / grid_step_deg) * grid_step_deg, 2)
    fov_q = round(float(fov_arcmin), 1)
    grid_key = f"ra{ra_q:.2f}_dec{dec_q:.2f}_fov{fov_q:.1f}"
    return grid_key, ra_q, dec_q


class SpatialSqliteCache:
    """Two-tier SQLite spatial cache with per-thread connection pooling and SLRU pruning."""

    def __init__(
        self,
        db_path: str | None = None,
        max_cache_bytes: int | None = None,
        max_memory_bytes: int | None = None,
        last_access_update_interval: float = LAST_ACCESS_UPDATE_INTERVAL_SECONDS,
        patch_ttl_seconds: float | None = None,
        popularity_half_life_seconds: float | None = None,
    ) -> None:
        """Initialize the SQLite spatial cache and run pending SQL migrations.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ``SQLITE_DB_PATH`` or ``/tmp/star_hunters_cache.db``.
            max_cache_bytes: Maximum combined byte budget for cached sky patches
                and catalog regions before SLRU eviction triggers.
            max_memory_bytes: Optional byte budget for the in-process LRU cache.
                Defaults to ``IN_MEMORY_CACHE_MB`` env var or 128 MB (``0`` disables).
            last_access_update_interval: Minimum seconds between SQLite
                ``last_accessed`` timestamp writes for the same row.
            patch_ttl_seconds: TTL in seconds for ``sky_patches`` and
                ``spectra_cache`` entries before revalidation (default 7 days).
            popularity_half_life_seconds: Exponential decay half-life for multi-user
                patch popularity ranking (default 7 days).
        """
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", "/tmp/star_hunters_cache.db")
        self.max_cache_bytes = _resolve_max_cache_bytes(max_cache_bytes)
        self.last_access_update_interval = max(0.0, float(last_access_update_interval))
        self.patch_ttl_seconds = _resolve_patch_ttl_seconds(patch_ttl_seconds)
        self._half_life_seconds = _resolve_popularity_half_life_seconds(
            popularity_half_life_seconds
        )
        self._decay_lambda = math.log(2.0) / self._half_life_seconds
        self._mem_cache = _MemoryLruCache(max_bytes=_resolve_max_memory_bytes(max_memory_bytes))
        self._visitors_lock = threading.Lock()
        self._mem_visitors: dict[str, bytes] = {}
        self._hmac_lock = threading.Lock()
        self._hmac_day = int(time.time() // 86400)
        self._hmac_secret = os.urandom(32)
        self._telemetry_lock = threading.Lock()
        self._telemetry_deltas: dict[str, int] = {m: 0 for m in TELEMETRY_METRICS}
        self._local = threading.local()
        self._pool_lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._finalizer = weakref.finalize(
            self,
            self._close_connections,
            self._connections,
            self._pool_lock,
        )
        self._init_db()

    @staticmethod
    def _close_connections(
        connections: list[sqlite3.Connection],
        lock: threading.Lock,
    ) -> None:
        """Close every pooled SQLite connection without holding a reference to ``self``."""
        with lock:
            conns = list(connections)
            connections.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        """Flush pending telemetry counters and close all pooled SQLite connections."""
        try:
            with self._telemetry_lock:
                has_pending = any(v > 0 for v in self._telemetry_deltas.values())
            if has_pending:
                with self.connection() as conn:
                    self._flush_telemetry_locked(conn)
        except sqlite3.Error:
            pass
        with self._pool_lock:
            self._local = threading.local()
            conns = list(self._connections)
            self._connections.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _get_thread_connection(self) -> sqlite3.Connection:
        """Return or open the current thread's pooled SQLite connection."""
        local = self._local
        conn: sqlite3.Connection | None = getattr(local, "conn", None)
        if conn is not None:
            return conn

        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        with self._pool_lock:
            self._connections.append(conn)
        local.conn = conn
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the current thread's pooled SQLite connection inside a transaction.

        Commits on normal exit and rolls back on exception without closing the
        pooled connection.

        Yields:
            Active thread-local :class:`sqlite3.Connection`.
        """
        conn = self._get_thread_connection()
        with conn:
            yield conn

    def _init_db(self) -> None:
        """Enable persistent WAL mode and apply pending numbered SQL migrations."""
        migration_files = sorted(
            _MIGRATIONS_DIR.glob("*.sql"),
            key=lambda p: int(p.stem.split("_", 1)[0]),
        )
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current_version >= SCHEMA_VERSION:
                return
            for sql_path in migration_files:
                version_num = int(sql_path.stem.split("_", 1)[0])
                if version_num <= current_version:
                    continue
                conn.executescript(sql_path.read_text(encoding="utf-8"))
                conn.execute(f"PRAGMA user_version = {version_num}")

    def _invalidate_grid_key_memory(self, grid_key: str) -> None:
        """Remove all in-process LRU cache entries associated with ``grid_key`` in O(1)."""
        self._mem_cache.delete_tag(grid_key)
        with self._visitors_lock:
            self._mem_visitors.pop(grid_key, None)

    def compute_visitor_token(
        self, client_ip: str | None, user_agent: str | None = None
    ) -> bytes | None:
        """Compute an anonymous 4-byte daily HMAC visitor token purely in RAM.

        Rotates the 32-byte random secret every UTC day without touching disk or
        setting browser cookies/localStorage IDs.
        """
        ip_norm = (client_ip or "").strip()
        ua_norm = (user_agent or "").strip()
        if not ip_norm and not ua_norm:
            return None
        today = int(time.time() // 86400)
        with self._hmac_lock:
            if today != self._hmac_day:
                self._hmac_day = today
                self._hmac_secret = os.urandom(32)
            secret = self._hmac_secret
        msg = f"{ip_norm or 'unknown'}|{ua_norm}".encode("utf-8", errors="replace")
        return hmac.new(secret, msg, hashlib.sha256).digest()[:4]

    def _incr_telemetry(self, metric: str, amount: int = 1) -> None:
        if amount <= 0 or metric not in self._telemetry_deltas:
            return
        with self._telemetry_lock:
            self._telemetry_deltas[metric] += int(amount)

    def record_client_cache_delta(self, header_value: str | None) -> None:
        """Parse ``X-Cache-Delta: p=<int>,c=<int>`` into aggregate telemetry deltas."""
        if not header_value:
            return
        for part in header_value.split(","):
            key, sep, val = part.strip().partition("=")
            if not sep:
                continue
            try:
                count = max(0, min(1000, int(val.strip())))
            except ValueError:
                continue
            if key.strip() == "p" and count > 0:
                self._incr_telemetry("client_local_patch_hits", count)
            elif key.strip() == "c" and count > 0:
                self._incr_telemetry("client_local_catalog_hits", count)

    def record_frame_serve(self, dtype_header: str) -> None:
        """Record whether a served detector frame used ``float32`` or ``float16``."""
        if dtype_header == "float32":
            self._incr_telemetry("float32_frame_serves", 1)
        elif dtype_header == "float16":
            self._incr_telemetry("float16_frame_serves", 1)

    def _flush_telemetry_locked(self, conn: sqlite3.Connection) -> None:
        """Upsert accumulated in-memory telemetry deltas into SQLite on an active write."""
        with self._telemetry_lock:
            pending = [(k, v) for k, v in self._telemetry_deltas.items() if v > 0]
            if not pending:
                return
            for k, _ in pending:
                self._telemetry_deltas[k] = 0
        conn.executemany(
            """
            INSERT INTO cache_telemetry_counters (metric, value)
            VALUES (?, ?)
            ON CONFLICT(metric) DO UPDATE SET value = value + excluded.value
            """,
            pending,
        )

    def _is_entry_expired(self, created_at: float, now: float) -> bool:
        """Return True when a cached row has exceeded ``self.patch_ttl_seconds``."""
        if self.patch_ttl_seconds <= 0.0:
            return True
        return (now - float(created_at)) > self.patch_ttl_seconds

    def _decode_ttl_mem_entry(
        self, cached_bytes: bytes, *, allow_stale: bool = False
    ) -> dict[str, Any] | None:
        """Decode a ``{"created_at", "payload"}`` RAM cache entry, honoring TTL."""
        wrapper = json.loads(cached_bytes)
        if isinstance(wrapper, dict) and "created_at" in wrapper and "payload" in wrapper:
            if not allow_stale and self._is_entry_expired(
                float(wrapper["created_at"]), time.time()
            ):
                return None
            payload: dict[str, Any] = wrapper["payload"]
            return payload
        return wrapper

    @staticmethod
    def _encode_ttl_mem_entry(created_at: float, payload: dict[str, Any]) -> bytes:
        """Serialize a ``{"created_at", "payload"}`` wrapper for ``_mem_cache``."""
        return json.dumps(
            {"created_at": float(created_at), "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8")

    def _get_json_row(
        self,
        table: Literal["sky_patches", "spectra_cache", "capital_zenith_cache"],
        json_column: Literal["manifest_json", "catalogs_json", "payload_json"],
        key_column: Literal["grid_key", "spectrum_key", "cache_key"],
        key_value: str,
        mem_key: str | None = None,
        *,
        enforce_ttl: bool = False,
        allow_stale: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a JSON payload from memory or SQLite and throttle ``last_accessed`` writes."""
        if mem_key is not None:
            cached_bytes = self._mem_cache.get(mem_key)
            if cached_bytes is not None:
                if enforce_ttl:
                    decoded = self._decode_ttl_mem_entry(cached_bytes, allow_stale=allow_stale)
                    if decoded is not None:
                        return decoded
                else:
                    return json.loads(cached_bytes)

        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT {json_column}, created_at, last_accessed "
                f"FROM {table} WHERE {key_column} = ?",
                (key_value,),
            ).fetchone()
            if not row:
                return None
            created_at = float(row["created_at"])
            if enforce_ttl and not allow_stale and self._is_entry_expired(created_at, now):
                return None
            if now - float(row["last_accessed"]) >= self.last_access_update_interval:
                conn.execute(
                    f"UPDATE {table} SET last_accessed = ? WHERE {key_column} = ?",
                    (now, key_value),
                )
            raw_json = str(row[json_column])
        payload: dict[str, Any] = json.loads(raw_json)
        if mem_key is not None:
            encoded = (
                self._encode_ttl_mem_entry(created_at, payload)
                if enforce_ttl
                else raw_json.encode("utf-8")
            )
            self._mem_cache.put(mem_key, encoded)
        return payload

    def peek_manifest_memory(
        self, grid_key: str, *, allow_stale: bool = False
    ) -> dict[str, Any] | None:
        """Fetch a sky-patch manifest solely from the in-process RAM LRU cache."""
        cached_bytes = self._mem_cache.get(f"manifest:{grid_key}")
        if cached_bytes is None:
            return None
        manifest = self._decode_ttl_mem_entry(cached_bytes, allow_stale=allow_stale)
        if manifest is None:
            return None
        manifest["query"]["cached"] = True
        return manifest

    def peek_catalogs_memory(
        self, grid_key: str, *, allow_stale: bool = False
    ) -> dict[str, Any] | None:
        """Fetch cached catalog overlays solely from the in-process RAM LRU cache."""
        cached_bytes = self._mem_cache.get(f"catalogs:{grid_key}")
        if cached_bytes is None:
            return None
        payload = self._decode_ttl_mem_entry(cached_bytes, allow_stale=allow_stale)
        if payload is None:
            return None
        payload["query"]["cached"] = True
        return payload

    def peek_spectrum_memory(
        self, spectrum_key: str, *, allow_stale: bool = False
    ) -> dict[str, Any] | None:
        """Fetch a cached 102-band spectrum solely from the in-process RAM LRU cache."""
        cached_bytes = self._mem_cache.get(f"spectrum:{spectrum_key}")
        if cached_bytes is None:
            return None
        return self._decode_ttl_mem_entry(cached_bytes, allow_stale=allow_stale)

    def peek_frame_gzip_memory(
        self,
        grid_key: str,
        filename: str,
        precision: Literal["float32", "float16"] = "float32",
    ) -> tuple[bytes, str] | None:
        """Fetch a pre-gzipped binary frame solely from the in-process RAM LRU cache."""
        if precision == "float16" and not filename.endswith("_flags.bin"):
            return None
        dtype_header = "uint8" if filename.endswith("_flags.bin") else "float32"
        cached_gz = self._mem_cache.get(f"frame_gz:{grid_key}:{filename}:{dtype_header}")
        if cached_gz is None:
            return None
        return cached_gz, dtype_header

    def get_manifest(self, grid_key: str, *, allow_stale: bool = False) -> dict[str, Any] | None:
        """Fetch a cached sky-patch manifest, rehydrating deduplicated catalog overlays.

        Args:
            grid_key: Quantized spatial grid identifier.
            allow_stale: When ``True``, return entries older than ``patch_ttl_seconds``.

        Returns:
            Manifest dictionary with ``query.cached = True``, or ``None``.
        """
        mem_key = f"manifest:{grid_key}"
        cached_bytes = self._mem_cache.get(mem_key)
        if cached_bytes is not None:
            manifest = self._decode_ttl_mem_entry(cached_bytes, allow_stale=allow_stale)
            if manifest is not None:
                manifest["query"]["cached"] = True
                return manifest

        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                "SELECT manifest_json, catalogs_json, created_at, last_accessed "
                "FROM sky_patches WHERE grid_key = ?",
                (grid_key,),
            ).fetchone()
            if not row:
                return None
            created_at = float(row["created_at"])
            if not allow_stale and self._is_entry_expired(created_at, now):
                return None
            if now - float(row["last_accessed"]) >= self.last_access_update_interval:
                conn.execute(
                    "UPDATE sky_patches SET last_accessed = ? WHERE grid_key = ?",
                    (now, grid_key),
                )
            manifest = json.loads(row["manifest_json"])
            if "catalog_overlays" not in manifest:
                catalogs_payload = json.loads(row["catalogs_json"])
                manifest["catalog_overlays"] = catalogs_payload["catalog_overlays"]
        manifest["query"]["cached"] = True
        self._mem_cache.put(mem_key, self._encode_ttl_mem_entry(created_at, manifest))
        return manifest

    def get_catalogs(self, grid_key: str, *, allow_stale: bool = False) -> dict[str, Any] | None:
        """Fetch cached catalog overlays for a sky patch and mark as cached.

        Args:
            grid_key: Quantized spatial grid identifier.
            allow_stale: When ``True``, return entries older than ``patch_ttl_seconds``.

        Returns:
            Catalogs dictionary with ``query.cached = True``, or ``None``.
        """
        payload = self._get_json_row(
            "sky_patches",
            "catalogs_json",
            "grid_key",
            grid_key,
            mem_key=f"catalogs:{grid_key}",
            enforce_ttl=True,
            allow_stale=allow_stale,
        )
        if payload is not None:
            payload["query"]["cached"] = True
        return payload

    def _read_encoded_frame_from_sqlite(
        self, grid_key: str, filename: str, encoding: str
    ) -> bytes | None:
        """Read stored pre-gzipped frame bytes from SQLite without touching ``_mem_cache``."""
        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT bf.compressed_blob, sp.last_accessed
                FROM binary_frames bf
                JOIN sky_patches sp ON sp.grid_key = bf.grid_key
                WHERE bf.grid_key = ? AND bf.filename = ? AND bf.encoding = ?
                """,
                (grid_key, filename, encoding),
            ).fetchone()
            if not row:
                return None
            if now - float(row["last_accessed"]) >= self.last_access_update_interval:
                conn.execute(
                    "UPDATE sky_patches SET last_accessed = ? WHERE grid_key = ?",
                    (now, grid_key),
                )
            return bytes(row["compressed_blob"])

    def _read_raw_frame_from_sqlite(self, grid_key: str, filename: str) -> bytes | None:
        """Read and decompress a binary frame to raw bytes without touching ``_mem_cache``."""
        encoding = "gzip_uint8" if filename.endswith("_flags.bin") else "gzip_float32"
        blob = self._read_encoded_frame_from_sqlite(grid_key, filename, encoding)
        if blob is None:
            return None
        return gzip.decompress(blob)

    def get_frame(self, grid_key: str, filename: str) -> bytes | None:
        """Fetch and decompress a cached binary frame buffer.

        Args:
            grid_key: Quantized spatial grid identifier.
            filename: Frame filename (e.g. ``"pass_1.bin"``).

        Returns:
            Decompressed binary frame bytes, or ``None`` if missing.
        """
        mem_key = f"frame_raw:{grid_key}:{filename}"
        cached_raw = self._mem_cache.get(mem_key)
        if cached_raw is not None:
            return cached_raw

        raw_bytes = self._read_raw_frame_from_sqlite(grid_key, filename)
        if raw_bytes is None:
            return None
        self._mem_cache.put(mem_key, raw_bytes)
        return raw_bytes

    def get_frame_gzip(
        self,
        grid_key: str,
        filename: str,
        precision: Literal["float32", "float16"] = "float32",
    ) -> tuple[bytes, str] | None:
        """Fetch a wire-ready gzipped binary frame from RAM or SQLite with zero recompression.

        Only canonical ``gzip_float32`` detector cubes and ``gzip_uint8`` flag masks
        are persisted; ``float16`` requests return ``None`` here so the service layer
        converts on-the-fly from ``float32`` without duplicating RAM/disk cache entries.

        Args:
            grid_key: Quantized spatial grid identifier.
            filename: Frame filename (e.g. ``"pass_1.bin"``).
            precision: Target floating-point precision for detector cubes.

        Returns:
            Tuple of ``(gzip_bytes, dtype_header)``, or ``None`` if missing or ``float16``.
        """
        if precision == "float16" and not filename.endswith("_flags.bin"):
            return None
        dtype_header = "uint8" if filename.endswith("_flags.bin") else "float32"
        target_encoding = "gzip_uint8" if dtype_header == "uint8" else "gzip_float32"
        gz_key = f"frame_gz:{grid_key}:{filename}:{dtype_header}"
        cached_gz = self._mem_cache.get(gz_key)
        if cached_gz is not None:
            return cached_gz, dtype_header

        blob = self._read_encoded_frame_from_sqlite(grid_key, filename, target_encoding)
        if blob is None:
            return None

        self._mem_cache.put(gz_key, blob)
        return blob, dtype_header

    def record_patch_hit(
        self, grid_key: str, visitor_token: bytes | None = None
    ) -> bool:
        """Record a sky-patch cache hit, deduplicating repeat visitors and updating popularity.

        Returns ``True`` when a new distinct visitor was recorded, or ``False`` for
        same-visitor repeats (or when ``visitor_token`` is absent).
        """
        if not visitor_token or len(visitor_token) != 4:
            self._incr_telemetry("same_user_backend_repeats", 1)
            return False

        with self._visitors_lock:
            cached_ring = self._mem_visitors.get(grid_key)
        if cached_ring is not None and _visitor_in_ring(cached_ring, visitor_token):
            self._incr_telemetry("same_user_backend_repeats", 1)
            return False

        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT distinct_visitors, popularity_score, last_popularity_ts,
                       recent_visitors, last_accessed
                FROM sky_patches
                WHERE grid_key = ?
                """,
                (grid_key,),
            ).fetchone()
            if not row:
                return False

            ring = bytes(row["recent_visitors"] or b"")
            if _visitor_in_ring(ring, visitor_token):
                with self._visitors_lock:
                    self._mem_visitors[grid_key] = ring
                self._incr_telemetry("same_user_backend_repeats", 1)
                if now - float(row["last_accessed"]) >= self.last_access_update_interval:
                    conn.execute(
                        "UPDATE sky_patches SET last_accessed = ? WHERE grid_key = ?",
                        (now, grid_key),
                    )
                    self._flush_telemetry_locked(conn)
                return False

            # distinct_visitors serves two roles:
            # 1) Eviction tier gate: patches with distinct_visitors < 2 stay in the
            #    probationary Pass 1 tier so single-user exploration never evicts
            #    shared patches (distinct_visitors >= 2).
            # 2) Deduplication gate: popularity_score only increments when a visitor's
            #    4-byte token is absent from the 128-slot recent_visitors ring buffer.
            #
            # Epoch-anchored log-decay algebra (eliminates background cleanup timers):
            # - At any future time t, a patch's decayed score is:
            #       S(t) = S_last * exp(-lambda * (t - t_last))
            # - Comparing two patches A and B at time t:
            #       S_A(t) < S_B(t)
            #   <=> ln(S_A) - lambda * (t - t_A) < ln(S_B) - lambda * (t - t_B)
            # - Adding lambda * (t - t0) to both sides cancels current time t completely:
            #       ln(S_A) + lambda * (t_A - t0) < ln(S_B) + lambda * (t_B - t0)
            # - Storing popularity_rank = ln(S) + lambda * (t_last - t0) therefore lets
            #   SQLite rank decayed popularity using a static B-tree index without ever
            #   waking up to decay stale rows.
            last_pop_ts = float(row["last_popularity_ts"] or row["last_accessed"] or now)
            dt = max(0.0, now - last_pop_ts)
            old_score = max(1e-9, float(row["popularity_score"] or 1.0))
            new_score = old_score * math.exp(-self._decay_lambda * dt) + 1.0
            new_rank = math.log(new_score) + self._decay_lambda * (now - POPULARITY_EPOCH_T0)
            new_ring = _append_visitor_ring(ring, visitor_token)
            new_distinct = int(row["distinct_visitors"] or 1) + 1

            conn.execute(
                """
                UPDATE sky_patches
                SET distinct_visitors = ?,
                    popularity_score = ?,
                    last_popularity_ts = ?,
                    popularity_rank = ?,
                    recent_visitors = ?,
                    last_accessed = ?
                WHERE grid_key = ?
                """,
                (new_distinct, new_score, now, new_rank, new_ring, now, grid_key),
            )
            self._incr_telemetry("common_multi_user_hits", 1)
            self._flush_telemetry_locked(conn)
            with self._visitors_lock:
                self._mem_visitors[grid_key] = new_ring
            return True

    def put_sky_patch(
        self,
        grid_key: str,
        ra_center: float,
        dec_center: float,
        fov_arcmin: float,
        manifest: dict[str, Any],
        frames: dict[str, bytes],
        visitor_token: bytes | None = None,
    ) -> None:
        """Persist a sky-patch manifest, catalog overlays, and ``float32``/``uint8`` frames.

        Stores only ``gzip_float32`` detector cubes (and ``gzip_uint8`` flag masks)
        to maximize SQLite capacity, populates ``_mem_cache`` immediately for follow-up
        ``/frames`` requests, and initializes SLRU popularity metadata.
        """
        now = time.time()
        slim_manifest = {key: value for key, value in manifest.items() if key != "catalog_overlays"}
        manifest_str = json.dumps(slim_manifest, separators=(",", ":"))
        catalogs_payload = {
            "query": manifest["query"],
            "wcs": manifest["wcs"],
            "catalog_overlays": manifest["catalog_overlays"],
        }
        catalogs_str = json.dumps(catalogs_payload, separators=(",", ":"))

        frame_rows: list[tuple[str, str, bytes, int, str]] = []
        mem_frame_entries: list[tuple[str, bytes]] = []
        total_bytes = len(manifest_str.encode("utf-8")) + len(catalogs_str.encode("utf-8"))
        for fname, raw_bytes in frames.items():
            if fname.endswith("_flags.bin"):
                comp = gzip.compress(raw_bytes, compresslevel=3)
                frame_rows.append((grid_key, fname, comp, len(raw_bytes), "gzip_uint8"))
                mem_frame_entries.append((f"frame_gz:{grid_key}:{fname}:uint8", comp))
                total_bytes += len(comp)
            elif len(raw_bytes) == BINARY_DETECTOR_FLOAT32_BYTES:
                gz32 = gzip.compress(raw_bytes, compresslevel=3)
                frame_rows.append((grid_key, fname, gz32, len(raw_bytes), "gzip_float32"))
                mem_frame_entries.append((f"frame_gz:{grid_key}:{fname}:float32", gz32))
                total_bytes += len(gz32)
            else:
                comp = gzip.compress(raw_bytes, compresslevel=3)
                frame_rows.append((grid_key, fname, comp, len(raw_bytes), "gzip_float32"))
                mem_frame_entries.append((f"frame_gz:{grid_key}:{fname}:float32", comp))
                total_bytes += len(comp)

        initial_visitors = (
            visitor_token if (visitor_token and len(visitor_token) == 4) else b""
        )
        # Initial popularity_rank = ln(1.0) + lambda * (now - t0) = lambda * (now - t0)
        initial_rank = self._decay_lambda * (now - POPULARITY_EPOCH_T0)

        with self.connection() as conn:
            conn.execute(
                "DELETE FROM sky_patches WHERE grid_key = ?",
                (grid_key,),
            )
            conn.execute(
                """
                INSERT INTO sky_patches (
                    grid_key, ra_center, dec_center, fov_arcmin,
                    manifest_json, catalogs_json, byte_size, created_at, last_accessed,
                    distinct_visitors, popularity_score, last_popularity_ts,
                    popularity_rank, recent_visitors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grid_key,
                    ra_center,
                    dec_center,
                    fov_arcmin,
                    manifest_str,
                    catalogs_str,
                    total_bytes,
                    now,
                    now,
                    1,
                    1.0,
                    now,
                    initial_rank,
                    initial_visitors,
                ),
            )
            if frame_rows:
                conn.executemany(
                    """
                    INSERT INTO binary_frames (
                        grid_key, filename, compressed_blob, raw_byte_size, encoding
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    frame_rows,
                )
            self._prune_if_needed(conn, preserve_grid_key=grid_key)
            self._incr_telemetry("unique_cold_misses", 1)
            self._flush_telemetry_locked(conn)

        self._invalidate_grid_key_memory(grid_key)
        # Populate _mem_cache immediately so follow-up /frames requests skip SQLite
        cached_manifest = copy.copy(manifest)
        cached_manifest["query"] = {**manifest["query"], "cached": True}
        self._mem_cache.put(
            f"manifest:{grid_key}", self._encode_ttl_mem_entry(now, cached_manifest)
        )
        cached_catalogs = {
            "query": {**manifest["query"], "cached": True},
            "wcs": manifest["wcs"],
            "catalog_overlays": manifest["catalog_overlays"],
        }
        self._mem_cache.put(
            f"catalogs:{grid_key}", self._encode_ttl_mem_entry(now, cached_catalogs)
        )
        for mem_key, blob in mem_frame_entries:
            self._mem_cache.put(mem_key, blob)
        with self._visitors_lock:
            self._mem_visitors[grid_key] = initial_visitors

    def get_patch_byte_size(self, grid_key: str) -> int:
        """Return the stored compressed byte size of a cached sky patch.

        Args:
            grid_key: Quantized spatial grid identifier.

        Returns:
            Stored byte size, or ``0`` if the patch is not cached.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT byte_size FROM sky_patches WHERE grid_key = ?",
                (grid_key,),
            ).fetchone()
            return int(row["byte_size"]) if row else 0

    def get_spectrum(
        self, spectrum_key: str, *, allow_stale: bool = False
    ) -> dict[str, Any] | None:
        """Fetch a cached 102-band spectrum payload.

        Args:
            spectrum_key: Cache key for the sky position's spectrum.
            allow_stale: When ``True``, return entries older than ``patch_ttl_seconds``.

        Returns:
            Cached spectrum payload dictionary, or ``None``.
        """
        return self._get_json_row(
            "spectra_cache",
            "payload_json",
            "spectrum_key",
            spectrum_key,
            mem_key=f"spectrum:{spectrum_key}",
            enforce_ttl=True,
            allow_stale=allow_stale,
        )

    def put_spectrum(
        self,
        spectrum_key: str,
        ra: float,
        dec: float,
        payload: dict[str, Any],
    ) -> None:
        """Persist a 102-band spectrum payload in SQLite.

        Args:
            spectrum_key: Cache key for the sky position's spectrum.
            ra: Right Ascension in degrees.
            dec: Declination in degrees.
            payload: Spectrum payload dictionary.
        """
        now = time.time()
        payload_str = json.dumps(payload)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO spectra_cache (
                    spectrum_key, ra, dec, payload_json, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (spectrum_key, ra, dec, payload_str, now, now),
            )
        self._mem_cache.delete(f"spectrum:{spectrum_key}")

    def get_zenith_response(self, cache_key: str) -> dict[str, Any] | None:
        """Fetch a cached Earth-to-Sky zenith lookup and mark it as cached.

        Args:
            cache_key: Quantized location/time zenith cache key.

        Returns:
            Zenith response dictionary with ``earth.cached = True``, or ``None``.
        """
        payload = self._get_json_row(
            "capital_zenith_cache",
            "payload_json",
            "cache_key",
            cache_key,
            mem_key=f"zenith:{cache_key}",
        )
        if payload is not None:
            payload["earth"]["cached"] = True
        return payload

    def put_zenith_response(
        self,
        cache_key: str,
        lat: float,
        lon: float,
        payload: dict[str, Any],
        city_slug: str | None = None,
    ) -> None:
        """Persist an Earth-to-Sky zenith response in SQLite.

        Args:
            cache_key: Quantized location/time zenith cache key.
            lat: Observer latitude in degrees.
            lon: Observer longitude in degrees.
            payload: Validated Earth-to-Sky response dictionary.
            city_slug: Optional matching capital city slug.
        """
        now = time.time()
        payload_str = json.dumps(payload)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO capital_zenith_cache (
                    cache_key, city_slug, lat, lon, payload_json, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cache_key, city_slug, lat, lon, payload_str, now, now),
            )
        self._mem_cache.delete(f"zenith:{cache_key}")

    @staticmethod
    def _catalog_region_key(
        source: str,
        ra: float,
        dec: float,
        radius_deg: float,
    ) -> tuple[str, float, float, float]:
        """Build a quantized cache key and bucket tuple for a catalog region query."""
        ra_b = quantize_bucket(ra, DEFAULT_GRID_STEP_DEG, wrap_360=True)
        dec_b = quantize_bucket(dec, DEFAULT_GRID_STEP_DEG, wrap_360=False)
        rad_b = round(max(0.01, float(radius_deg)), 3)
        key = f"{source}:ra{ra_b:.2f}:dec{dec_b:.2f}:r{rad_b:.3f}"
        return key, ra_b, dec_b, rad_b

    def get_cached_catalog_region(
        self,
        source: str,
        ra: float,
        dec: float,
        radius_deg: float,
        allow_stale: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch cached SIMBAD/Gaia/MOST region payload if present and valid.

        Args:
            source: Catalog source identifier (e.g. ``"simbad"``, ``"gaia_dr3"``).
            ra: Right Ascension center in degrees.
            dec: Declination center in degrees.
            radius_deg: Search radius in degrees.
            allow_stale: If True, return expired rows marked with ``stale=True``.

        Returns:
            Cached catalog region dictionary, or ``None`` if absent or expired.
        """
        now = time.time()
        region_key, _, _, _ = self._catalog_region_key(source, ra, dec, radius_deg)
        mem_key = f"catreg:{region_key}"
        cached_mem = self._mem_cache.get(mem_key)
        if cached_mem is not None:
            wrapper = json.loads(cached_mem)
            is_expired = float(wrapper["expires_at"]) < now
            if is_expired and not allow_stale:
                return None
            payload: dict[str, Any] = wrapper["payload"]
            payload["cache_hit"] = True
            payload["stale"] = is_expired
            return payload

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT payload_json, expires_at, last_accessed
                FROM catalog_region_cache
                WHERE region_key = ?
                """,
                (region_key,),
            ).fetchone()
            if not row:
                return None
            expires_at = float(row["expires_at"])
            is_expired = expires_at < now
            if is_expired and not allow_stale:
                return None
            if now - float(row["last_accessed"]) >= self.last_access_update_interval:
                conn.execute(
                    "UPDATE catalog_region_cache SET last_accessed = ? WHERE region_key = ?",
                    (now, region_key),
                )
            payload = json.loads(row["payload_json"])
        wrapper_bytes = json.dumps(
            {"expires_at": expires_at, "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8")
        self._mem_cache.put(mem_key, wrapper_bytes)
        payload["cache_hit"] = True
        payload["stale"] = is_expired
        return payload

    def put_cached_catalog_region(
        self,
        source: str,
        ra: float,
        dec: float,
        radius_deg: float,
        payload: dict[str, Any],
        ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
    ) -> None:
        """Persist a quantized catalog query response in SQLite with TTL + LRU pruning.

        Args:
            source: Catalog source identifier.
            ra: Right Ascension center in degrees.
            dec: Declination center in degrees.
            radius_deg: Search radius in degrees.
            payload: JSON-serializable catalog response payload.
            ttl_seconds: Cache time-to-live in seconds (minimum 60s).
        """
        now = time.time()
        expires_at = now + max(60.0, float(ttl_seconds))
        region_key, ra_b, dec_b, rad_b = self._catalog_region_key(source, ra, dec, radius_deg)
        payload_str = json.dumps(payload, separators=(",", ":"))
        byte_size = len(payload_str.encode("utf-8"))
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO catalog_region_cache (
                    region_key, source, ra_bucket, dec_bucket, radius_bucket,
                    payload_json, byte_size, fetched_at, expires_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    region_key,
                    source,
                    ra_b,
                    dec_b,
                    rad_b,
                    payload_str,
                    byte_size,
                    now,
                    expires_at,
                    now,
                ),
            )
            self._prune_if_needed(conn)
        self._mem_cache.delete(f"catreg:{region_key}")

    def upsert_ephemeris_points(self, points: list[dict[str, Any]]) -> int:
        """Upsert JPL Horizons / MOST rows indexed by ``(date_bucket, ra_bucket, dec_bucket)``.

        Args:
            points: List of ephemeris point dictionaries containing ``naif_id``,
                ``body_name``, ``mjd``, ``ra``, and ``dec``.

        Returns:
            Number of rows upserted.
        """
        if not points:
            return 0
        now = time.time()
        rows_to_insert: list[tuple[Any, ...]] = []
        for pt in points:
            ra_val = float(pt["ra"]) % 360.0
            dec_val = max(-90.0, min(90.0, float(pt["dec"])))
            mjd_val = float(pt["mjd"])
            date_bucket = str(pt.get("date_bucket") or mjd_to_date_bucket(mjd_val))
            ra_b = quantize_bucket(ra_val, DEFAULT_GRID_STEP_DEG, wrap_360=True)
            dec_b = quantize_bucket(dec_val, DEFAULT_GRID_STEP_DEG, wrap_360=False)
            vmag_val = float(pt["vmag"]) if pt.get("vmag") is not None else None
            rows_to_insert.append(
                (
                    str(pt["naif_id"]),
                    str(pt["body_name"]),
                    str(pt.get("body_class", "solar_system")),
                    date_bucket,
                    mjd_val,
                    ra_val,
                    dec_val,
                    vmag_val,
                    ra_b,
                    dec_b,
                    str(pt.get("source", "jpl_horizons")),
                    float(pt.get("fetched_at", now)),
                )
            )
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO solar_system_ephemeris (
                    naif_id, body_name, body_class, date_bucket, mjd,
                    ra, dec, vmag, ra_bucket, dec_bucket, source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
        return len(rows_to_insert)

    def get_ephemeris_for_body(
        self,
        naif_id: str,
        mjd: float,
    ) -> dict[str, Any] | None:
        """Return interpolated cached ephemeris for ``naif_id`` near ``mjd``.

        Args:
            naif_id: NAIF / Horizons body identifier.
            mjd: Target Modified Julian Date.

        Returns:
            Interpolated ephemeris dictionary, or ``None`` if unavailable.
        """
        target_mjd = float(mjd)
        buckets = (
            mjd_to_date_bucket(target_mjd - 1.0),
            mjd_to_date_bucket(target_mjd),
            mjd_to_date_bucket(target_mjd + 1.0),
        )
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT naif_id, body_name, body_class, date_bucket, mjd, ra, dec, vmag, source
                FROM solar_system_ephemeris
                WHERE naif_id = ?
                  AND date_bucket IN (?, ?, ?)
                ORDER BY mjd ASC
                """,
                (str(naif_id), *buckets),
            ).fetchall()
        if not rows:
            return None
        interp_ra, interp_dec, interp_vmag = interpolate_ephemeris_points(rows, target_mjd)
        first = rows[0]
        return {
            "naif_id": str(first["naif_id"]),
            "body_name": str(first["body_name"]),
            "body_class": str(first["body_class"]),
            "mjd": target_mjd,
            "ra": round(interp_ra, 6),
            "dec": round(interp_dec, 6),
            "vmag": round(interp_vmag, 3) if interp_vmag is not None else None,
            "source": str(first["source"]),
        }

    def query_ephemeris_in_fov(
        self,
        mjd: float,
        ra: float,
        dec: float,
        radius_deg: float,
    ) -> list[dict[str, Any]]:
        """Query Solar System bodies in or traversing the FOV around ``mjd``.

        Fetches candidate rows within a padded RA/Dec bounding box across
        ``mjd +/- 1 day`` and delegates interpolation and angular-separation
        filtering to :func:`app.utils.astronomy.filter_ephemeris_candidates_in_fov`.

        Args:
            mjd: Target Modified Julian Date.
            ra: Field center Right Ascension in degrees.
            dec: Field center Declination in degrees.
            radius_deg: Search radius in degrees.

        Returns:
            List of matching Solar System body dictionaries sorted by angular
            separation.
        """
        target_mjd = float(mjd)
        date_prev = mjd_to_date_bucket(target_mjd - 1.0)
        date_curr = mjd_to_date_bucket(target_mjd)
        date_next = mjd_to_date_bucket(target_mjd + 1.0)

        norm_ra, clamped_dec, dec_min, dec_max, ra_low, ra_high, wraps = (
            compute_ephemeris_fov_bounds(ra, dec, radius_deg)
        )
        if ra_low is None or ra_high is None:
            ra_clause = "1 = 1"
            ra_params: tuple[Any, ...] = ()
        elif not wraps:
            ra_clause = "ra_bucket BETWEEN ? AND ?"
            ra_params = (ra_low, ra_high)
        else:
            ra_clause = "(ra_bucket >= ? OR ra_bucket <= ?)"
            ra_params = (ra_low, ra_high)

        sql = f"""
            SELECT naif_id, body_name, body_class, date_bucket, mjd, ra, dec, vmag, source
            FROM solar_system_ephemeris
            WHERE date_bucket IN (?, ?, ?)
              AND dec_bucket BETWEEN ? AND ?
              AND {ra_clause}
            ORDER BY naif_id ASC, mjd ASC
        """
        params: tuple[Any, ...] = (date_prev, date_curr, date_next, dec_min, dec_max, *ra_params)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return filter_ephemeris_candidates_in_fov(
            rows=rows,
            target_mjd=target_mjd,
            ra=norm_ra,
            dec=clamped_dec,
            radius_deg=radius_deg,
        )

    def get_ephemeris_stats(self) -> dict[str, Any]:
        """Return count of stored ephemeris rows, bodies, and cached catalog regions.

        Returns:
            Summary statistics dictionary for ephemeris and catalog caches.
        """
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COUNT(DISTINCT naif_id) AS body_count,
                       MIN(date_bucket) AS min_date,
                       MAX(date_bucket) AS max_date
                FROM solar_system_ephemeris
                """
            ).fetchone()
            cat_row = conn.execute(
                """
                SELECT COUNT(*) AS region_count,
                       COALESCE(SUM(byte_size), 0) AS catalog_bytes
                FROM catalog_region_cache
                """
            ).fetchone()
        return {
            "ephemeris_rows": int(row["row_count"]) if row else 0,
            "distinct_bodies": int(row["body_count"]) if row else 0,
            "min_date": row["min_date"] if row else None,
            "max_date": row["max_date"] if row else None,
            "catalog_regions": int(cat_row["region_count"]) if cat_row else 0,
            "catalog_bytes": int(cat_row["catalog_bytes"]) if cat_row else 0,
        }

    def _prune_if_needed(
        self,
        conn: sqlite3.Connection,
        preserve_grid_key: str | None = None,
    ) -> int:
        """Evict expired catalog entries and LRU patches until within ``max_cache_bytes``."""
        row = conn.execute(
            "SELECT COALESCE(SUM(byte_size), 0) AS total_bytes FROM sky_patches"
        ).fetchone()
        cat_row = conn.execute(
            "SELECT COALESCE(SUM(byte_size), 0) AS total_bytes FROM catalog_region_cache"
        ).fetchone()
        patch_bytes = int(row["total_bytes"]) if row else 0
        cat_bytes = int(cat_row["total_bytes"]) if cat_row else 0
        total_bytes = patch_bytes + cat_bytes
        evicted = 0
        if total_bytes <= self.max_cache_bytes:
            return evicted

        # Pass 0: Purge expired catalog_region_cache rows first, then oldest catalog rows if needed
        if cat_bytes > 0:
            now = time.time()
            expired = conn.execute(
                "SELECT region_key, byte_size FROM catalog_region_cache WHERE expires_at < ?",
                (now,),
            ).fetchall()
            for r in expired:
                conn.execute(
                    "DELETE FROM catalog_region_cache WHERE region_key = ?",
                    (r["region_key"],),
                )
                self._mem_cache.delete(f"catreg:{r['region_key']}")
                total_bytes -= int(r["byte_size"])
                evicted += 1
            if total_bytes > self.max_cache_bytes:
                cat_rows = conn.execute(
                    "SELECT region_key, byte_size FROM catalog_region_cache "
                    "ORDER BY last_accessed ASC, rowid ASC"
                ).fetchall()
                for r in cat_rows:
                    if total_bytes <= self.max_cache_bytes:
                        break
                    conn.execute(
                        "DELETE FROM catalog_region_cache WHERE region_key = ?",
                        (r["region_key"],),
                    )
                    self._mem_cache.delete(f"catreg:{r['region_key']}")
                    total_bytes -= int(r["byte_size"])
                    evicted += 1

        if total_bytes <= self.max_cache_bytes:
            return evicted

        # Segmented SLRU eviction using the idx_sky_patches_eviction B-tree index:
        # - Pass 1 (distinct_visitors < 2 -> key 0): Single-user exploration patches
        #   are evicted first in pure LRU order (last_accessed ASC).
        # - Pass 2 (distinct_visitors >= 2 -> key 1): Shared multi-user patches are
        #   evicted by popularity_rank ASC, which equals ln(S(t)) + lambda*(t - t0)
        #   shifted by a global constant, so older unvisited popular patches decay
        #   below active ones without any background cleanup job.
        cursor = conn.execute(
            """
            SELECT grid_key, byte_size
            FROM sky_patches
            ORDER BY (CASE WHEN distinct_visitors >= 2 THEN 1 ELSE 0 END) ASC,
                     (CASE WHEN distinct_visitors >= 2 THEN popularity_rank ELSE last_accessed END) ASC,
                     last_accessed ASC,
                     rowid ASC
            """
        )
        rows = cursor.fetchall()
        for r in rows:
            if total_bytes <= self.max_cache_bytes:
                break
            if preserve_grid_key and r["grid_key"] == preserve_grid_key:
                continue
            conn.execute("DELETE FROM sky_patches WHERE grid_key = ?", (r["grid_key"],))
            self._invalidate_grid_key_memory(str(r["grid_key"]))
            total_bytes -= int(r["byte_size"])
            evicted += 1
        return evicted

    def clear(self) -> None:
        """Delete all cached entries and telemetry counters across every SQLite table."""
        with self._telemetry_lock:
            for metric in self._telemetry_deltas:
                self._telemetry_deltas[metric] = 0
        with self.connection() as conn:
            conn.execute("DELETE FROM binary_frames")
            conn.execute("DELETE FROM sky_patches")
            conn.execute("DELETE FROM spectra_cache")
            conn.execute("DELETE FROM capital_zenith_cache")
            conn.execute("DELETE FROM catalog_region_cache")
            conn.execute("DELETE FROM solar_system_ephemeris")
            conn.execute("DELETE FROM cache_telemetry_counters")
        self._mem_cache.clear()
        with self._visitors_lock:
            self._mem_visitors.clear()

    def get_cache_stats(self, top_n: int = 10) -> dict[str, Any]:
        """Return aggregate cache telemetry ratios, storage occupancy, and top patches."""
        now = time.time()
        with self.connection() as conn:
            self._flush_telemetry_locked(conn)
            counter_rows = conn.execute(
                "SELECT metric, value FROM cache_telemetry_counters"
            ).fetchall()
            patch_agg = conn.execute(
                """
                SELECT COUNT(*) AS cnt,
                       COALESCE(SUM(CASE WHEN distinct_visitors >= 2 THEN 1 ELSE 0 END), 0) AS multi_cnt,
                       COALESCE(SUM(byte_size), 0) AS total_bytes
                FROM sky_patches
                """
            ).fetchone()
            cat_agg = conn.execute(
                "SELECT COALESCE(SUM(byte_size), 0) AS total_bytes FROM catalog_region_cache"
            ).fetchone()
            top_rows = conn.execute(
                """
                SELECT grid_key, distinct_visitors, popularity_score, last_popularity_ts,
                       byte_size, last_accessed
                FROM sky_patches
                ORDER BY (CASE WHEN distinct_visitors >= 2 THEN 1 ELSE 0 END) DESC,
                         (CASE WHEN distinct_visitors >= 2 THEN popularity_rank ELSE last_accessed END) DESC,
                         last_accessed DESC
                LIMIT ?
                """,
                (max(1, int(top_n)),),
            ).fetchall()

        counters = {m: 0 for m in TELEMETRY_METRICS}
        for row in counter_rows:
            name = str(row["metric"])
            if name in counters:
                counters[name] = int(row["value"])

        local_patch = counters["client_local_patch_hits"]
        common_hits = counters["common_multi_user_hits"]
        repeat_backend = counters["same_user_backend_repeats"]
        cold_misses = counters["unique_cold_misses"]
        total_patch_events = local_patch + common_hits + repeat_backend + cold_misses

        def _pct(num: int, den: int) -> float:
            return round((num / den) * 100.0, 2) if den > 0 else 0.0

        top_patches = []
        for r in top_rows:
            dt = max(0.0, now - float(r["last_popularity_ts"] or r["last_accessed"] or now))
            decayed = float(r["popularity_score"] or 1.0) * math.exp(-self._decay_lambda * dt)
            top_patches.append(
                {
                    "grid_key": str(r["grid_key"]),
                    "distinct_visitors": int(r["distinct_visitors"] or 1),
                    "decayed_popularity": round(decayed, 4),
                    "byte_size": int(r["byte_size"] or 0),
                }
            )

        patch_count = int(patch_agg["cnt"]) if patch_agg else 0
        multi_count = int(patch_agg["multi_cnt"]) if patch_agg else 0
        return {
            "telemetry": {
                **counters,
                "total_patch_events": total_patch_events,
                "common_multi_user_pct": _pct(common_hits, total_patch_events),
                "same_user_local_pct": _pct(local_patch, total_patch_events),
                "same_user_backend_pct": _pct(repeat_backend, total_patch_events),
                "unique_cold_miss_pct": _pct(cold_misses, total_patch_events),
                "client_absorption_pct": _pct(local_patch, local_patch + repeat_backend),
            },
            "storage": {
                "ram_used_bytes": self._mem_cache.current_bytes,
                "ram_max_bytes": self._mem_cache.max_bytes,
                "sqlite_patch_count": patch_count,
                "sqlite_multi_user_patches": multi_count,
                "sqlite_single_user_patches": max(0, patch_count - multi_count),
                "sqlite_patch_bytes": int(patch_agg["total_bytes"]) if patch_agg else 0,
                "sqlite_catalog_bytes": int(cat_agg["total_bytes"]) if cat_agg else 0,
                "sqlite_max_bytes": self.max_cache_bytes,
            },
            "top_patches": top_patches,
        }
