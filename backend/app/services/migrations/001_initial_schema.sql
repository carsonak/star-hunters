-- Quantized SQLite spatial cache schema (SCHEMA_VERSION = 1)

CREATE TABLE IF NOT EXISTS sky_patches (
    grid_key TEXT PRIMARY KEY,
    ra_center REAL NOT NULL,
    dec_center REAL NOT NULL,
    fov_arcmin REAL NOT NULL,
    manifest_json TEXT NOT NULL,
    catalogs_json TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sky_patches_last_accessed
ON sky_patches(last_accessed);

CREATE TABLE IF NOT EXISTS binary_frames (
    grid_key TEXT NOT NULL,
    filename TEXT NOT NULL,
    encoding TEXT NOT NULL,
    compressed_blob BLOB NOT NULL,
    raw_byte_size INTEGER NOT NULL,
    PRIMARY KEY (grid_key, filename, encoding),
    FOREIGN KEY (grid_key) REFERENCES sky_patches(grid_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS spectra_cache (
    spectrum_key TEXT PRIMARY KEY,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS capital_zenith_cache (
    cache_key TEXT PRIMARY KEY,
    city_slug TEXT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog_region_cache (
    region_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    ra_bucket REAL NOT NULL,
    dec_bucket REAL NOT NULL,
    radius_bucket REAL NOT NULL,
    payload_json TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_catalog_region_lookup
ON catalog_region_cache(source, ra_bucket, dec_bucket, radius_bucket);

CREATE INDEX IF NOT EXISTS idx_catalog_region_last_accessed
ON catalog_region_cache(last_accessed);

CREATE TABLE IF NOT EXISTS solar_system_ephemeris (
    naif_id TEXT NOT NULL,
    body_name TEXT NOT NULL,
    body_class TEXT NOT NULL,
    date_bucket TEXT NOT NULL,
    mjd REAL NOT NULL,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    vmag REAL,
    ra_bucket REAL NOT NULL,
    dec_bucket REAL NOT NULL,
    source TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    PRIMARY KEY (naif_id, date_bucket)
);

CREATE INDEX IF NOT EXISTS idx_ephem_lookup
ON solar_system_ephemeris(date_bucket, ra_bucket, dec_bucket);
