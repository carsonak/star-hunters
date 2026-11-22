#!/usr/bin/env node
/**
 * Client-side two-phase multi-configuration benchmark & Fly.io $5 Ceiling Calculator:
 * - Simulates 60 virtual user sessions per phase (20 conn/min x 3 min) with Quick, Standard,
 *   and Deep explorer personas matching run_benchmark.py.
 * - Tests multiple ClientPatchCache capacities (64, 128, 192 entries).
 * - Exercises decodeDetectorBinaryBuffer (float16), renderEpochBuffersToRgba (256x256),
 *   detectAndTrackBodiesAtClick, and projectSkySin / unprojectSkySin.
 * - Generates benchmarks/results/benchmark_report.md with side-by-side backend configuration
 *   comparisons and a multi-configuration Fly.io $5/month ceiling sustainability matrix.
 */

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import { ClientPatchCache, gridCacheKey } from '../frontend/src/services/clientPatchCache.ts';
import { decodeDetectorBinaryBuffer } from '../frontend/src/utils/binaryDecoder.ts';
import { detectAndTrackBodiesAtClick } from '../frontend/src/utils/clickTracker.ts';
import { renderEpochBuffersToRgba } from '../frontend/src/utils/pixelEngine.ts';
import { BACKGROUND_STARS, projectSkySin, unprojectSkySin } from '../frontend/src/utils/skyProjection.ts';
import { computeGridKey, computeViewportViewBox } from '../frontend/src/utils/wcsViewport.ts';

const BENCH_DIR = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = path.join(BENCH_DIR, 'results');
const RESULTS_PATH = path.join(RESULTS_DIR, 'benchmark_results.json');
const REPORT_PATH = path.join(RESULTS_DIR, 'benchmark_report.md');

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function round2(v) {
  return Math.round(v * 100) / 100;
}

function percentile(values, pct) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * (pct / 100);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
}

function makeSyntheticFloat16Buffer(seedOffset = 0) {
  const buf = new ArrayBuffer(786432);
  const words = new Uint16Array(buf);
  words.fill(0x2800);
  for (let d = 0; d < 6; d += 1) {
    const base = d * 65536;
    const cx = 128 + seedOffset;
    const cy = 128;
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) {
        words[base + (cy + dy) * 256 + (cx + dx)] = 0x4800;
      }
    }
  }
  return buf;
}

function makeManifest(ra, dec, gridKey) {
  return {
    query: { ra, dec, fov_arcmin: 5.0, grid_key: gridKey, cached: false },
    target_metadata: {
      name: `Patch ${gridKey}`,
      slug: gridKey,
      category: 'custom',
      description: 'Benchmark patch',
      constellation: 'Orion',
    },
    data_source_summary: {
      primary_mission: 'SPHEREx',
      has_multi_epoch_spherex: true,
      uses_wise_fallback: false,
      baseline_days: 182.5,
      status_note: '2 SPHEREx comparison epoch(s)',
      provenance_tier: 'primary',
      survey_name: 'SPHEREx DR1',
    },
    wcs: {
      width: 256,
      height: 256,
      crval: [ra, dec],
      crpix: [128.5, 128.5],
      cdelt: [-(5.0 / 60.0) / 256.0, (5.0 / 60.0) / 256.0],
    },
    epochs: [
      {
        epoch_id: 'epoch_1',
        label: 'Pass 1',
        mjd: 60500.0,
        iso_date: '2024-07-09T00:00:00Z',
        mission: 'SPHEREx',
        detectors_data_url: '',
        flags_mask_url: '',
        available_detectors: [1, 2, 3, 4, 5, 6],
      },
      {
        epoch_id: 'epoch_2',
        label: 'Pass 2',
        mjd: 60682.5,
        iso_date: '2025-01-07T12:00:00Z',
        mission: 'SPHEREx',
        detectors_data_url: '',
        flags_mask_url: '',
        available_detectors: [1, 2, 3, 4, 5, 6],
      },
    ],
    catalog_overlays: { known_objects: [], gaia_dr3: [], most_solar_system: [] },
  };
}

const HOT_COORDS = [
  [133.7951, -7.2451], [269.4521, 4.6934], [162.328, -53.319], [282.4557, -23.8362],
  [217.4289, -62.6795], [346.6224, -27.542], [116.1, -30.5], [10.6847, 41.2687],
  [83.8221, -5.3911], [187.7059, 12.3911], [201.697, -47.4795], [266.4168, -29.0078],
  [279.2347, 38.7837], [310.358, 45.28], [344.4127, 20.7688], [148.8882, 69.0653],
  [83.6331, 22.0145], [202.4696, 47.1952], [322.4932, 12.167], [56.75, 24.1167],
];

const CITIES_POOL_COORDS = Array.from({ length: 60 }, (_, i) => [
  Number(((i * 6.0 + 3.0) % 360).toFixed(4)),
  Number((-60 + (i % 25) * 5.0).toFixed(4)),
]);

function summarizeLat(arr) {
  if (arr.length === 0) return { count: 0, p50: 0, p95: 0, p99: 0, max: 0 };
  return {
    count: arr.length,
    p50: round2(percentile(arr, 50)),
    p95: round2(percentile(arr, 95)),
    p99: round2(percentile(arr, 99)),
    max: round2(Math.max(...arr)),
  };
}

async function runClientPhase(phaseName, seed, cache, totalSessions = 60) {
  const rng = mulberry32(seed);
  const f16Pass1 = makeSyntheticFloat16Buffer(0);
  const f16Pass2 = makeSyntheticFloat16Buffer(3);
  const rgbaOut = new Uint8ClampedArray(256 * 256 * 4);

  const mockObjects = Array.from({ length: 200 }, (_, i) => ({
    ra: (i * 1.79) % 360,
    dec: -70 + (i % 140),
  }));

  const statsBefore = cache.getStats();
  const cpuBefore = process.cpuUsage();
  const wallStart = performance.now();

  const decodeMs = [];
  const renderMs = [];
  const globeFrameMs = [];
  const trackMs = [];

  for (let session = 0; session < totalSessions; session += 1) {
    const roll = rng();
    const isQuick = roll < 0.25;
    const isDeep = roll >= 0.75;
    const globeFrames = isQuick ? 15 : isDeep ? 30 : 22;
    const coldCount = isDeep ? 2 : 1;
    const hotCount = isQuick ? 3 : isDeep ? 5 : 4;
    const cityCount = isQuick ? 1 : 2;
    const sliderSweeps = isQuick ? 12 : isDeep ? 20 : 16;

    // 1. Globe panning & zooming
    let ra = rng() * 360;
    let dec = -60 + rng() * 120;
    let fov = 70;
    for (let f = 0; f < globeFrames; f += 1) {
      ra = (ra + 4.5) % 360;
      fov = Math.max(1.6, fov * 0.88);
      const t0 = performance.now();
      for (const star of BACKGROUND_STARS) {
        projectSkySin(star.ra, star.dec, ra, dec, fov, 1280, 800);
      }
      for (const obj of mockObjects) {
        projectSkySin(obj.ra, obj.dec, ra, dec, fov, 1280, 800);
      }
      unprojectSkySin(640, 400, ra, dec, fov, 1280, 800);
      globeFrameMs.push(performance.now() - t0);
    }

    // 2. Patch visits (cold + hot + city zeniths)
    const targets = [];
    for (let c = 0; c < coldCount; c += 1) {
      targets.push([Number((rng() * 359).toFixed(4)), Number((-70 + rng() * 140).toFixed(4))]);
    }
    for (let h = 0; h < hotCount; h += 1) {
      targets.push(HOT_COORDS[Math.floor(rng() * HOT_COORDS.length)]);
    }
    for (let z = 0; z < cityCount; z += 1) {
      targets.push(CITIES_POOL_COORDS[Math.floor(rng() * CITIES_POOL_COORDS.length)]);
    }

    for (const [tra, tdec] of targets) {
      const gk = gridCacheKey(computeGridKey(tra, tdec, 5.0));
      let bundle = await cache.getPatch(gk);
      if (!bundle) {
        const td0 = performance.now();
        const planes1 = decodeDetectorBinaryBuffer(f16Pass1, 'float16');
        const planes2 = decodeDetectorBinaryBuffer(f16Pass2, 'float16');
        decodeMs.push(performance.now() - td0);

        bundle = {
          cacheKey: gk,
          manifest: makeManifest(tra, tdec, gk),
          epochBuffers: [planes1, planes2],
          flagsBuffers: [new Uint8Array(65536), new Uint8Array(65536)],
          spectrum: null,
          storedAt: Date.now(),
          lastAccessed: Date.now(),
        };
        await cache.putPatch(bundle);
      }

      const modes = ['blink', 'subtract', 'coadd'];
      for (let step = 0; step < sliderSweeps; step += 1) {
        const minLam = 0.75 + (step % 6) * 0.5;
        const maxLam = Math.min(5.0, minLam + 1.5);
        computeViewportViewBox(1.0 + (step % 4) * 1.5, (step - 6) * 4, 0);
        const tr0 = performance.now();
        renderEpochBuffersToRgba({
          epochBuffers: bundle.epochBuffers,
          flagsBuffers: bundle.flagsBuffers,
          activeEpochIndex: step % 2,
          comparisonMode: modes[step % 3],
          contrastStretch: 'asinh',
          wavelengthRangeUm: [minLam, maxLam],
          passAIndex: 0,
          passBIndex: 1,
          outBuffer: rgbaOut,
        });
        renderMs.push(performance.now() - tr0);
      }

      const tt0 = performance.now();
      detectAndTrackBodiesAtClick({
        clickX: 128,
        clickY: 128,
        originEpochIndex: 0,
        epochBuffers: bundle.epochBuffers,
        manifest: bundle.manifest,
        wavelengthRangeUm: [0.75, 5.0],
      });
      trackMs.push(performance.now() - tt0);
    }
  }

  const wallMs = performance.now() - wallStart;
  const cpuUsed = process.cpuUsage(cpuBefore);
  const statsAfter = cache.getStats();
  const memNow = process.memoryUsage();

  const hits = statsAfter.patchHits - statsBefore.patchHits;
  const misses = statsAfter.patchMisses - statsBefore.patchMisses;
  const totalLookups = hits + misses;

  return {
    phase: phaseName,
    seed,
    sessions: totalSessions,
    wall_ms: round2(wallMs),
    client_cpu_user_ms: round2(cpuUsed.user / 1000),
    client_cpu_system_ms: round2(cpuUsed.system / 1000),
    client_patch_cache: {
      capacity_entries: statsAfter.maxEntries,
      resident_entries_after: statsAfter.memoryPatches,
      hits,
      misses,
      hit_rate_pct: totalLookups > 0 ? round2((hits / totalLookups) * 100) : 0,
    },
    v8_memory_mb: {
      rss_mb: round2(memNow.rss / (1024 * 1024)),
      heap_used_mb: round2(memNow.heapUsed / (1024 * 1024)),
      external_mb: round2(memNow.external / (1024 * 1024)),
      array_buffers_mb: round2(memNow.arrayBuffers / (1024 * 1024)),
    },
    operations_ms: {
      decode_float16_cube: summarizeLat(decodeMs),
      render_rgba_256x256: summarizeLat(renderMs),
      globe_sin_projection_frame: summarizeLat(globeFrameMs),
      click_to_track: summarizeLat(trackMs),
    },
  };
}

const FLY_CANDIDATE_CONFIGS = [
  { label: 'shared-cpu-2x (2048 MB RAM) + 1 GB Vol', compute_usd_hr: 0.0245, vol_gb: 1 },
  { label: 'shared-cpu-2x (2048 MB RAM) + 2 GB Vol', compute_usd_hr: 0.0245, vol_gb: 2 },
  { label: 'shared-cpu-2x (4096 MB RAM) + 2 GB Vol', compute_usd_hr: 0.0384, vol_gb: 2 },
  { label: 'shared-cpu-2x (4096 MB RAM) + 4 GB Vol', compute_usd_hr: 0.0384, vol_gb: 4 },
  { label: 'shared-cpu-4x (4096 MB RAM) + 1 GB Vol (Current)', compute_usd_hr: 0.0490, vol_gb: 1 },
  { label: 'shared-cpu-4x (4096 MB RAM) + 2 GB Vol', compute_usd_hr: 0.0490, vol_gb: 2 },
  { label: 'shared-cpu-4x (4096 MB RAM) + 3 GB Vol', compute_usd_hr: 0.0490, vol_gb: 3 },
  { label: 'shared-cpu-4x (4096 MB RAM) + 4 GB Vol', compute_usd_hr: 0.0490, vol_gb: 4 },
  { label: 'shared-cpu-4x (8192 MB RAM) + 4 GB Vol', compute_usd_hr: 0.0768, vol_gb: 4 },
  { label: 'performance-2x (4096 MB Dedicated) + 3 GB Vol', compute_usd_hr: 0.1240, vol_gb: 3 },
];

export function computeFlyFiveDollarMatrix(wireMbPerSession) {
  const sessionsPerHour = 1200;
  const fullEgressGbPerHour = (wireMbPerSession * sessionsPerHour) / 1024.0;
  const egressRateUsdPerGb = 0.02;

  return FLY_CANDIDATE_CONFIGS.map((cfg) => {
    const volUsdMo = round2(cfg.vol_gb * 0.15);
    const availBudgetUsd = Math.max(0, 5.0 - volUsdMo);

    const demoMaxHrsMo = round2(availBudgetUsd / cfg.compute_usd_hr);
    const demoHrsPerDay = round2(demoMaxHrsMo / 30.0);

    const hrCost0Pct = cfg.compute_usd_hr + fullEgressGbPerHour * egressRateUsdPerGb;
    const peakHrs0Pct = round2(availBudgetUsd / hrCost0Pct);
    const peakSessions0Pct = Math.floor(peakHrs0Pct * sessionsPerHour);

    const hrCost50Pct = cfg.compute_usd_hr + fullEgressGbPerHour * 0.5 * egressRateUsdPerGb;
    const peakHrs50Pct = round2(availBudgetUsd / hrCost50Pct);
    const peakSessions50Pct = Math.floor(peakHrs50Pct * sessionsPerHour);

    const hrCost80Pct = cfg.compute_usd_hr + fullEgressGbPerHour * 0.2 * egressRateUsdPerGb;
    const peakHrs80Pct = round2(availBudgetUsd / hrCost80Pct);
    const peakSessions80Pct = Math.floor(peakHrs80Pct * sessionsPerHour);

    return {
      ...cfg,
      vol_usd_mo: volUsdMo,
      avail_budget_usd: round2(availBudgetUsd),
      full_egress_gb_hr: round2(fullEgressGbPerHour),
      demo_max_hrs_mo: demoMaxHrsMo,
      demo_hrs_per_day: demoHrsPerDay,
      peak_20cpm_0pct_cache: { hrs: peakHrs0Pct, sessions: peakSessions0Pct, usd_hr: round2(hrCost0Pct) },
      peak_20cpm_50pct_cache: { hrs: peakHrs50Pct, sessions: peakSessions50Pct, usd_hr: round2(hrCost50Pct) },
      peak_20cpm_80pct_cache: { hrs: peakHrs80Pct, sessions: peakSessions80Pct, usd_hr: round2(hrCost80Pct) },
    };
  });
}

function summarizeTopLevelTiers(tierOutcomes) {
  let ram = 0;
  let sqlite = 0;
  let miss = 0;
  for (const [k, v] of Object.entries(tierOutcomes)) {
    if (k === 'frame_raw:sqlite_disk') continue;
    if (k.endsWith(':in_memory_lru')) ram += v;
    else if (k.endsWith(':sqlite_disk')) sqlite += v;
    else if (k.endsWith(':miss')) miss += v;
  }
  const total = Math.max(1, ram + sqlite + miss);
  return {
    ram,
    sqlite,
    miss,
    ram_pct: ((ram / total) * 100).toFixed(1),
    sqlite_pct: ((sqlite / total) * 100).toFixed(1),
    combined_hit_pct: (((ram + sqlite) / total) * 100).toFixed(1),
  };
}

function buildReportMarkdown(data) {
  const cfgRuns = data.configuration_runs ?? [
    {
      config: {
        name: 'default',
        fly_equivalent: 'shared-cpu',
        vcpus: data.backend_run_1?.server_cpu?.pinned_vcpus ?? '-',
        coadd_workers: data.backend_run_1?.server_cpu?.coadd_workers ?? '-',
        in_memory_cache_mb: data.backend_run_1?.application_caches?.mem_lru_max_mb ?? 512,
        sqlite_max_cache_mb: data.backend_run_1?.disk_and_sqlite?.sqlite_max_mb ?? 750,
      },
      backend_run_1: data.backend_run_1,
      backend_run_2: data.backend_run_2,
    },
  ];

  const lines = [
    '# Multi-Configuration 3-Minute (20 Conn/Min, 60-140s Stay Time) Benchmark Report',
    '',
    'All backend simulations ran locally at **20 connections/minute (`60` virtual user sessions arriving over `180s` per phase, max `25` concurrent users, `60s-140s` staying time across Quick, Standard, and Deep explorer personas)**.',
    '- **Run 1 (`seed=42`):** Cold start (empty `_MemoryLruCache` and empty SQLite database).',
    '- **Run 2 (`seed=1337`):** Prewarmed follow-up reusing Run 1’s populated `_MemoryLruCache` and SQLite database with refreshed randomness seeds.',
    '',
    '## 1. Multi-Configuration Backend Resource & Cache Comparison',
    '',
    '| Configuration (vCPUs / Coadd Workers / RAM LRU / SQLite Cap) | Phase | Drain Time (s) | All Reqs Mean / Weighted p50 / p95 (ms) | Heavy Reqs Mean / Weighted p50 / p95 (ms) | CPU Avg / Peak (Allocated vCPUs %) | Peak RSS (MB) | RAM LRU Used / Cap (MB) | SQLite End / Cap (MB) | Disk R / W (MB) | Wire Egress (MB/user · Mbps) | Top-Level Cache Hit % (RAM / SQLite / Combined) |',
    '| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ];

  for (const entry of cfgRuns) {
    const c = entry.config;
    for (const [label, run] of [
      ['Run 1 (Cold)', entry.backend_run_1],
      ['Run 2 (Prewarmed)', entry.backend_run_2],
    ]) {
      const tiers = summarizeTopLevelTiers(run.application_caches.tier_outcomes_delta);
      const cpuAvg = run.server_cpu.avg_pct_allocated_vcpus ?? '-';
      const cpuPeak = run.server_cpu.peak_pct_allocated_vcpus ?? '-';
      const allLat = run.overall_latency_ms ?? { mean_ms: 0, weighted_p50_ms: 0, weighted_p95_ms: 0 };
      const hvyLat = run.heavy_endpoints_latency_ms ?? { mean_ms: 0, weighted_p50_ms: 0, weighted_p95_ms: 0 };
      const cw = c.coadd_workers ?? run.server_cpu?.coadd_workers ?? '-';
      lines.push(
        `| **${c.name}** (${c.vcpus} vCPU / ${cw} coadd / ${c.in_memory_cache_mb}MB / ${c.sqlite_max_cache_mb}MB) | ${label} | ${run.elapsed_s}s | **${allLat.mean_ms}** / ${allLat.weighted_p50_ms} / ${allLat.weighted_p95_ms} ms | **${hvyLat.mean_ms}** / ${hvyLat.weighted_p50_ms} / ${hvyLat.weighted_p95_ms} ms | ${cpuAvg}% / ${cpuPeak}% | ${run.server_ram.peak_rss_mb} MB | ${run.application_caches.mem_lru_used_mb_after} / ${run.application_caches.mem_lru_max_mb} MB | ${run.disk_and_sqlite.end_db_mb} / ${c.sqlite_max_cache_mb} MB | ${run.disk_and_sqlite.io_read_mb} / ${run.disk_and_sqlite.io_write_mb} MB | ${run.network.mb_per_session_wire} MB · ${run.network.avg_egress_mbps} Mbps | ${tiers.ram_pct}% / ${tiers.sqlite_pct}% / **${tiers.combined_hit_pct}%** |`,
      );
    }
  }

  for (const entry of cfgRuns) {
    const c = entry.config;
    const b1 = entry.backend_run_1;
    const b2 = entry.backend_run_2;
    lines.push(
      '',
      `## 2. Endpoint Latencies for \`${c.name}\` (${c.fly_equivalent})`,
      '',
      '| Endpoint Group | Run 1 Cold Mean / p50 / p95 / p99 (ms) | Run 2 Prewarmed Mean / p50 / p95 / p99 (ms) | Errors (R1 / R2) |',
      '| :--- | ---: | ---: | ---: |',
    );
    const allEps = new Set([...Object.keys(b1.endpoints), ...Object.keys(b2.endpoints)]);
    for (const ep of [...allEps].sort()) {
      const e1 = b1.endpoints[ep] ?? { mean_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0, errors: 0 };
      const e2 = b2.endpoints[ep] ?? { mean_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0, errors: 0 };
      lines.push(
        `| \`${ep}\` | **${e1.mean_ms ?? '-'}** / ${e1.p50_ms} / ${e1.p95_ms} / ${e1.p99_ms} | **${e2.mean_ms ?? '-'}** / ${e2.p50_ms} / ${e2.p95_ms} / ${e2.p99_ms} | ${e1.errors} / ${e2.errors} |`,
      );
    }
  }

  const clientMatrix = data.client_capacity_matrix ?? [];
  if (clientMatrix.length > 0) {
    lines.push(
      '',
      `## 3. Client-Side Engine & \`ClientPatchCache\` Capacity Sweep (Node ${process.version})`,
      '',
      '| `ClientPatchCache` Capacity | Run 1 Hit Rate | Run 2 Hit Rate | V8 Heap / ArrayBuffers (MB) | `decodeFloat16` p50/p95 (ms) | `renderRgba` p50/p95 (ms) | `projectSkySin` p50/p95 (ms) | `clickToTrack` p50/p95 (ms) |',
      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    );
    for (const cm of clientMatrix) {
      const r1 = cm.client_run_1;
      const r2 = cm.client_run_2;
      lines.push(
        `| **${cm.capacity} entries** | ${r1.client_patch_cache.hits}/${r1.client_patch_cache.hits + r1.client_patch_cache.misses} (**${r1.client_patch_cache.hit_rate_pct}%**) | ${r2.client_patch_cache.hits}/${r2.client_patch_cache.hits + r2.client_patch_cache.misses} (**${r2.client_patch_cache.hit_rate_pct}%**) | ${r2.v8_memory_mb.heap_used_mb} / ${r2.v8_memory_mb.array_buffers_mb} MB | ${r2.operations_ms.decode_float16_cube.p50} / ${r2.operations_ms.decode_float16_cube.p95} | ${r2.operations_ms.render_rgba_256x256.p50} / ${r2.operations_ms.render_rgba_256x256.p95} | ${r2.operations_ms.globe_sin_projection_frame.p50} / ${r2.operations_ms.globe_sin_projection_frame.p95} | ${r2.operations_ms.click_to_track.p50} / ${r2.operations_ms.click_to_track.p95} |`,
      );
    }
  }

  const primaryB2 = cfgRuns[cfgRuns.length - 1].backend_run_2;
  const wireMbPerSession = primaryB2?.network?.mb_per_session_wire ?? 5.5;
  const flyMatrix = data.fly_five_dollar_ceiling ?? computeFlyFiveDollarMatrix(wireMbPerSession);

  lines.push(
    '',
    `## 4. Fly.io $5/Month Ceiling Sustainability Across Configurations (at Measured ${wireMbPerSession} MB/Session Wire Egress)`,
    '',
    '- **Continuous 20 Connections/Minute Peak (`1,200 sessions/hour`)** transfers **`' +
      round2((wireMbPerSession * 1200) / 1024) +
      ' GB/hour`** if 100% of visitors have cold browser caches (`@ $0.02/GB` Fly.io outbound transfer), or proportionally less as repeat visitors hit `ClientPatchCache` (IndexedDB + in-memory) and HTTP `Cache-Control: immutable` frame headers.',
    '',
    '| Fly.io VM + Volume Configuration | Compute (`$/hr`) | Fixed Vol (`$/mo`) | Budget Left Under $5 | Warm Demo Uptime (Max `hrs/mo` · `hrs/day`) | Continuous 20 Conn/Min Peak: 0% Browser Cache (`hrs` · `sessions`) | Continuous 20 Conn/Min Peak: 50% Browser Cache (`hrs` · `sessions`) | Continuous 20 Conn/Min Peak: 80% Browser Cache (`hrs` · `sessions`) |',
    '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  );

  for (const row of flyMatrix) {
    lines.push(
      `| **${row.label}** | $${row.compute_usd_hr.toFixed(4)}/hr | $${row.vol_usd_mo.toFixed(2)} | $${row.avail_budget_usd.toFixed(2)} | **${row.demo_max_hrs_mo} hrs** (${row.demo_hrs_per_day} h/day) | **${row.peak_20cpm_0pct_cache.hrs} hrs** (${row.peak_20cpm_0pct_cache.sessions.toLocaleString()} users @ $${row.peak_20cpm_0pct_cache.usd_hr}/hr) | **${row.peak_20cpm_50pct_cache.hrs} hrs** (${row.peak_20cpm_50pct_cache.sessions.toLocaleString()} users @ $${row.peak_20cpm_50pct_cache.usd_hr}/hr) | **${row.peak_20cpm_80pct_cache.hrs} hrs** (${row.peak_20cpm_80pct_cache.sessions.toLocaleString()} users @ $${row.peak_20cpm_80pct_cache.usd_hr}/hr) |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

async function main() {
  const isSmoke = process.argv.includes('--smoke-test');
  const totalSessions = isSmoke ? 4 : 60;
  const capacities = isSmoke ? [64] : [64, 128, 192];

  const clientCapacityMatrix = [];
  for (const cap of capacities) {
    const cache = new ClientPatchCache(cap);
    const r1 = await runClientPhase(`client_run_1_cold_cap${cap}`, 42, cache, totalSessions);
    const r2 = await runClientPhase(`client_run_2_prewarmed_cap${cap}`, 1337, cache, totalSessions);
    clientCapacityMatrix.push({ capacity: cap, client_run_1: r1, client_run_2: r2 });
  }

  const primaryClient = clientCapacityMatrix[Math.min(1, clientCapacityMatrix.length - 1)];
  if (isSmoke) {
    console.log('Client smoke test ok:', primaryClient.client_run_2.client_patch_cache);
    return;
  }

  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const existing = fs.existsSync(RESULTS_PATH)
    ? JSON.parse(fs.readFileSync(RESULTS_PATH, 'utf-8'))
    : {};
  const wireMbPerSession = existing.backend_run_2?.network?.mb_per_session_wire ?? 5.5;
  const flyFiveDollarCeiling = computeFlyFiveDollarMatrix(wireMbPerSession);

  const merged = {
    ...existing,
    client_run_1: primaryClient.client_run_1,
    client_run_2: primaryClient.client_run_2,
    client_capacity_matrix: clientCapacityMatrix,
    fly_five_dollar_ceiling: flyFiveDollarCeiling,
  };
  fs.writeFileSync(RESULTS_PATH, JSON.stringify(merged, null, 2), 'utf-8');
  if (merged.backend_run_1 && merged.backend_run_2) {
    fs.writeFileSync(REPORT_PATH, buildReportMarkdown(merged), 'utf-8');
  }
  console.log('Client multi-capacity benchmark & Fly $5 sustainability matrix completed:', {
    capacities_tested: capacities,
    wire_mb_per_session: wireMbPerSession,
  });
}

await main();
