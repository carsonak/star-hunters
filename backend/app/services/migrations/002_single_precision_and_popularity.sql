-- Migration 002: float32-only storage, 128-slot visitor deduplication,
-- epoch-anchored exponential popularity decay, and aggregate telemetry counters.

ALTER TABLE sky_patches ADD COLUMN distinct_visitors INTEGER NOT NULL DEFAULT 1;
ALTER TABLE sky_patches ADD COLUMN popularity_score REAL NOT NULL DEFAULT 1.0;
ALTER TABLE sky_patches ADD COLUMN last_popularity_ts REAL NOT NULL DEFAULT 0.0;
ALTER TABLE sky_patches ADD COLUMN popularity_rank REAL NOT NULL DEFAULT 0.0;
ALTER TABLE sky_patches ADD COLUMN recent_visitors BLOB NOT NULL DEFAULT X'';

-- Backfill existing rows using 7-day half-life (ln(2)/604800) anchored at t0=1700000000.0
UPDATE sky_patches
SET last_popularity_ts = last_accessed,
    popularity_rank = (0.6931471805599453 / 604800.0) * (last_accessed - 1700000000.0)
WHERE last_popularity_ts = 0.0;

-- Drop precomputed float16 blobs; float16 is now generated on-the-fly from float32
DELETE FROM binary_frames WHERE encoding = 'gzip_float16';

-- Recompute patch byte_size after dropping gzip_float16 rows
UPDATE sky_patches
SET byte_size = LENGTH(CAST(manifest_json AS BLOB))
    + LENGTH(CAST(catalogs_json AS BLOB))
    + COALESCE(
        (SELECT SUM(LENGTH(bf.compressed_blob))
         FROM binary_frames bf
         WHERE bf.grid_key = sky_patches.grid_key),
        0
    );

DROP INDEX IF EXISTS idx_sky_patches_last_accessed;

CREATE INDEX IF NOT EXISTS idx_sky_patches_eviction
ON sky_patches (
    (CASE WHEN distinct_visitors >= 2 THEN 1 ELSE 0 END) ASC,
    (CASE WHEN distinct_visitors >= 2 THEN popularity_rank ELSE last_accessed END) ASC,
    last_accessed ASC
);

CREATE TABLE IF NOT EXISTS cache_telemetry_counters (
    metric TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
