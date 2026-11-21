# Star Hunters Benchmarks & Stress-Test Suite

This directory (maintained on the fork) contains reproducible backend, client-engine, and cache-saturation stress benchmarks for **Star Hunters** (`FastAPI` + `SpatialSqliteCache` + `React/TypeScript` pixel & astrometry engines).

## Prerequisites

1. **Backend Python environment** (from the repository root):
   ```bash
   cd backend
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   cd ..
   ```
   *(If you need `httpx` for the stress/benchmark driver and it is not already installed, run `backend/.venv/bin/pip install httpx`.)*

2. **Node.js 22+** (required only for the TypeScript client-side engine benchmark `run_client_benchmark.mjs`, which uses native Node TypeScript stripping):
   ```bash
   node -v  # v22+ or v24+
   ```

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

Outputs are written to `benchmarks/benchmark_results.json`.

---

## 2. Client Engine & Fly.io `$5/Month` Ceiling Calculator (`run_client_benchmark.mjs`)

Benchmarks `ClientPatchCache` hit rates (`64`, `128`, `192` entry capacities), `decodeDetectorBinaryBuffer` (`float16`), `renderEpochBuffersToRgba`, `projectSkySin`, and `detectAndTrackBodiesAtClick`, and combines them with `benchmark_results.json` to produce a multi-configuration Fly.io `$5/month` sustainability report:

```bash
node benchmarks/run_client_benchmark.mjs
```

Outputs are written to `benchmarks/benchmark_report.md` and `benchmarks/benchmark_results.json`.

---

## 3. 6×4 Step-Load & Cache-Saturation Stress Test (`generate_stress_dataset.py` + `run_stress_test.py`)

Measures raw server throughput, latency percentiles (`mean`, `p50`, `p95`, `p99`), CPU/RSS/disk I/O, and cache eviction behavior with **zero artificial client sleeps** across **6 concurrency stages** (`C = 2, 6, 12, 20, 30, 40`) × **4 cache saturation tiers** (`50%`, `85%`, `100%`, `125%` of SQLite capacity).

### Step A: Generate the Realistic Sky Dataset & Action Library

Each wire-ready `gzip_float16` sky patch occupies **`~1.23 MB`** in SQLite. You can size the dataset by SQLite cap (`125%` auto-sizing), explicit target MB, or explicit patch count:

```bash
# Default: 125% of 4,096 MB SQLite cap (~5,120 MB / 4,163 unique sky patches)
backend/.venv/bin/python benchmarks/generate_stress_dataset.py --sqlite-max-mb 4096

# Or specify an exact target size or exact patch count:
# backend/.venv/bin/python benchmarks/generate_stress_dataset.py --target-size-mb 5120
# backend/.venv/bin/python benchmarks/generate_stress_dataset.py --num-patches 5000
```

This creates `benchmarks/stress_dataset/` containing `sky_patches_catalog.json`, `cutout_templates.npz`, and `action_library.json` (`670` unique parameterized actions across `12` weighted user-flow archetypes).

### Step B: Run the Stress Test

```bash
# Quick smoke check (C = 2, 4 at 50%, 100% with 2 users/tier)
backend/.venv/bin/python benchmarks/run_stress_test.py --smoke-test

# Full 6x4 matrix (C = 2,6,12,20,30,40 x tiers 50%,85%,100%,125%, 120 users/tier)
backend/.venv/bin/python benchmarks/run_stress_test.py \
  --in-memory-mb 1536 --sqlite-max-mb 4096
```

Useful flags:
- `--auto-clamp-wsl`: Clamps `in-memory-mb` and `sqlite-max-mb` to safe host headroom on low-RAM laptops.
- `--vcpus <N>`: Pin and size dynamic coadd workers to `<N>` cores (default: all host cores).
- `--stages "2,6,12,20"` / `--tiers "50,100"` / `--users-per-tier 60`: Customize the matrix dimensions.
- `--no-reset-between-stages`: Carry SQLite/RAM state across concurrency stages instead of resetting before each concurrency level.

Outputs are written to `benchmarks/stress_test_results.json` and `benchmarks/stress_test_report.md`.

