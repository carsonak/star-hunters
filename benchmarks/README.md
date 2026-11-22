# Star Hunters Benchmarks & Stress-Test Suite

This directory (maintained on the fork) contains reproducible backend, client-engine, and cache-saturation stress benchmarks for **Star Hunters** (`FastAPI` + `SpatialSqliteCache` + `React/TypeScript` pixel & astrometry engines).

## Directory Layout

- `lib/` — Shared modular helpers (`paths.py`, `host.py`, `telemetry.py`, `instrumentation.py`, `http_load.py`).
- `results/` *(git-ignored)* — Generated JSON metrics, Markdown reports, and server `.log` files.
- `data/` *(git-ignored)* — Generated SQLite databases (`bench_cache.db`, `stress_cache.db`), prewarm snapshots, and `stress_dataset/`.

## Prerequisites

1. **Backend Python environment** (from the repository root):
   ```bash
   cd backend
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   cd ..
   ```
   *(If `httpx` is not already installed in your environment, run `backend/.venv/bin/pip install httpx`.)*

2. **Node.js 22+** (required only for the TypeScript client-side engine benchmark `run_client_benchmark.mjs`, which uses native Node TypeScript stripping):
   ```bash
   node -v  # v22+
   ```

3. **Alternate backend target (optional):**
   By default, the Python scripts test `<repo>/backend`. To benchmark a separate backend directory or worktree, pass `--backend-dir <path>` or set `STAR_HUNTERS_BACKEND_DIR=<path>`.

---

## 1. Multi-Configuration 3-Minute Backend Benchmark (`run_benchmark.py`)

Simulates realistic user waves arriving at **20 connections/minute** over **180 seconds** (`60` sessions per phase, `60s–140s` staying time, capped at `25` concurrent users) across two phases:
- **Run 1 (Cold):** Starts with an empty SQLite + RAM cache (`0` prewarmed entries).
- **Run 2 (Prewarmed):** Prewarms showcase targets and runs against the populated cache.

```bash
# Inspect available host RAM, disk, and recommended resource matrix
backend/.venv/bin/python benchmarks/run_benchmark.py --probe-resources

# Run the full multi-configuration matrix (or pass --configs "name:vcpus:coadd_workers:ram_mb:sqlite_mb")
backend/.venv/bin/python benchmarks/run_benchmark.py --matrix
```

Outputs are written to `benchmarks/results/benchmark_results.json` and `benchmarks/results/bench_server_<config>.log`, with SQLite databases stored in `benchmarks/data/bench_cache.db`.

---

## 2. Client Engine & Fly.io `$5/Month` Ceiling Calculator (`run_client_benchmark.mjs`)

Benchmarks `ClientPatchCache` hit rates (`64`, `128`, `192` entry capacities), `decodeDetectorBinaryBuffer` (`float16`), `renderEpochBuffersToRgba`, `projectSkySin`, and `detectAndTrackBodiesAtClick`, and combines them with `benchmarks/results/benchmark_results.json` to produce a multi-configuration Fly.io `$5/month` sustainability report:

```bash
node benchmarks/run_client_benchmark.mjs
```

Outputs are written to `benchmarks/results/benchmark_report.md` and `benchmarks/results/benchmark_results.json`.

---

## 3. 5×5 Step-Load & Cache-Saturation Stress Test (`generate_stress_dataset.py` + `run_stress_test.py`)

Measures raw server throughput, latency percentiles (`mean`, `p50`, `p95`, `p99`), CPU/RSS/disk I/O, and cache eviction behavior with **zero artificial client sleeps** across **5 concurrency stages** (`C = 2, 6, 15, 25, 40`) × **5 cache saturation tiers** (`ram_75`, `ram_97`, `db_75`, `db_97`, `db_125`).
- `ram_75`, `ram_97`: Zero-eviction baselines sized to `75%` and `97%` of `in_memory_mb`.
- `db_75`, `db_97`: Disk-backed SQLite tiers sized to `75%` and `97%` of `sqlite_max_mb` (RAM evicts only).
- `db_125`: Overflow eviction tier at `125%` of `sqlite_max_mb` (both RAM and SQLite evictions).

### Step A: Generate the Realistic Sky Dataset & Action Library

Each wire-ready `gzip_float16` sky patch occupies **`~1.23 MB`** in SQLite. You can size the dataset by SQLite cap (`125%` auto-sizing), explicit target MB, or explicit patch count:

```bash
# Default proportional local test: 125% of 512 MB SQLite cap (~640 MB / 520 unique sky patches)
backend/.venv/bin/python benchmarks/generate_stress_dataset.py --sqlite-max-mb 512

# Or size for full 4 GB Fly.io SQLite cap / custom target MB / exact patch count:
# backend/.venv/bin/python benchmarks/generate_stress_dataset.py --sqlite-max-mb 4096
# backend/.venv/bin/python benchmarks/generate_stress_dataset.py --target-size-mb 5120
# backend/.venv/bin/python benchmarks/generate_stress_dataset.py --num-patches 5000
```

This creates `benchmarks/data/stress_dataset/` containing `sky_patches_catalog.json`, `cutout_templates.npz`, and `action_library.json` (`670` unique parameterized actions across `12` weighted user-flow archetypes).

### Step B: Run the Stress Test

```bash
# Quick smoke check (C = 2, 4 at ram_75, db_125 with snapshot pre-warming)
backend/.venv/bin/python benchmarks/run_stress_test.py --smoke-test

# Full 5x5 matrix (C = 2,6,15,25,40 x ram_75,ram_97,db_75,db_97,db_125, 50 users/tier)
backend/.venv/bin/python benchmarks/run_stress_test.py \
  --in-memory-mb 192 --sqlite-max-mb 512
```

Useful flags:
- `--ram-ratio <float>`: Derive `in-memory-mb` automatically from `sqlite-max-mb * ram-ratio` (e.g., `--ram-ratio 0.25`, `0.375`, `0.50`) to sweep optimal RAM:SQLite ratios.
- `--auto-clamp`: Clamps `in-memory-mb` and `sqlite-max-mb` to safe host headroom on memory- or disk-constrained machines.
- `--vcpus <N>`: Pin and size dynamic coadd workers (`N - max(1, N // 3)`) to `<N>` cores (default: all host cores).
- `--stages "2,6,15"` / `--tiers "ram_75,db_97,db_125"` / `--users-per-tier 80`: Customize matrix dimensions.
- `--no-prewarm`: Start each concurrency stage from a completely cold empty DB instead of restoring the `ram_75` snapshot.
- `--no-reset-between-stages`: Carry SQLite/RAM state across concurrency stages instead of restoring the snapshot before each concurrency level.
- `--backend-dir <path>`: Point the stress server at an alternate backend checkout.

Outputs are written to `benchmarks/results/stress_test_results.json`, `benchmarks/results/stress_test_report.md`, and `benchmarks/results/stress_server.log`, with SQLite databases stored under `benchmarks/data/`.
