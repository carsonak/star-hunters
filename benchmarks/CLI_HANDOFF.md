# Antigravity CLI Handoff Guide — 6×4 Step-Load Stress Test & Benchmarks

> **Purpose:** The user closes Antigravity IDE (which consumes significant WSL RAM) and runs the full stress-test matrix via Antigravity CLI (`agy`). Because Antigravity IDE and Antigravity CLI do not share conversation contexts, this guide bridges both agents.

---

## 1. Completed Backend Optimizations (`star-hunters-main/backend/` & `fly.toml`)

1. **Wire-Ready `gzip_float16` / `gzip_uint8` SQLite Storage (`spatial_cache.py`):**
   - `put_sky_patch()` pre-converts detector frames to `float16`, compresses with `gzip`, and stores wire-ready `gzip_float16` (`uint8` flags as `gzip_uint8`) directly in `binary_frames`.
   - Serving a `precision=float16` frame from SQLite after a RAM LRU eviction or machine suspend/resume now requires **zero** `zlib.decompress`, `float32 -> float16` conversion, or `gzip.compress` CPU work (~`1.23 MB/patch` measured on disk vs `~2.58 MB/patch` before, with backwards-compatible decoding to `float32` if requested).
2. **Clean I/O vs. CPU ThreadPool Isolation (`backend_service.py`, `main.py`):**
   - Dedicated `_cpu_executor` (for archival coadd + cold spectrum synthesis, gated cleanly on the event loop via per-loop `asyncio.Semaphore`) and `_io_executor` (for fast SQLite reads, zenith lookups, catalogs, and compression persistence).
   - Non-blocking `get_frame_bytes_async()` serves `_MemoryLruCache` RAM hits in `~2 µs` on the event loop and offloads SQLite disk reads to `_io_executor`.
   - Monotonic `compute_max_concurrent_coadds`: `<= 6` cores reserves `ceil(N / 2)`; `> 6` cores reserves `max(3, ceil(N / 4))` (minimum `3` cores for I/O/`asyncio` across `6–12` cores).
3. **Fly.io Sweet-Spot Concurrency & Per-IP Cold-Miss Rate Limiter (`fly.toml`, `backend_service.py`, `main.py`):**
   - `[http_service.concurrency]` set to `type = "connections"`, `soft_limit = 15`, `hard_limit = 25`.
   - Per-IP 60s sliding-window cold-miss guard (`COLD_MISS_RATE_LIMIT_PER_IP_PER_MIN`, default `30/min`; set `0` in benchmarks) returns `HTTP 429` + `Retry-After` if one IP floods uncached cold synthesis, while never throttling cached reads.

---

## 2. Regenerate the Stress Dataset (Configurable `--num-patches` / `--target-size-mb`)

With wire-ready `gzip_float16` storage (`~1.23 MB/patch` measured), `125%` of our `4,096 MB` SQLite cap equals `~5,120 MB` (`4,163` unique sky patches):

```bash
# Default: auto-sizes to 125% of --sqlite-max-mb (4,163 patches = ~5,120 MB)
/home/line/hackathons/star-hunters-workspace/star-hunters-main/backend/.venv/bin/python \
  /home/line/hackathons/star-hunters-workspace/benchmarks/generate_stress_dataset.py --sqlite-max-mb 4096

# Or specify exact target size or exact patch count:
#   --target-size-mb 5120
#   --num-patches 4500
```

---

## 3. Run the 6×4 Step-Load & Cache-Saturation Stress Test (`C = 2..40`, `50%/85%/100%/125%`)

Each of the **6 concurrency stages** (`C = 2, 6, 12, 20, 30, 40`) starts with a fresh reset cache and sweeps **4 cache saturation tiers** (`50%, 85%, 100%, 125%`) with **`120` unique spawned users per saturation tier** (`480` users/stage, `2,880` users across all `24` stage×saturation cells):

```bash
/home/line/hackathons/star-hunters-workspace/star-hunters-main/backend/.venv/bin/python \
  /home/line/hackathons/star-hunters-workspace/benchmarks/run_stress_test.py \
  --in-memory-mb 1536 --sqlite-max-mb 4096
```

*(Add `--auto-clamp-wsl` on low-RAM hosts, or customize with `--stages "2,6,12,20,30,40" --tiers "50,85,100,125" --users-per-tier 120`. Outputs are written to `/home/line/hackathons/star-hunters-workspace/benchmarks/stress_test_results.json` and `/home/line/hackathons/star-hunters-workspace/benchmarks/stress_test_report.md`.)*

