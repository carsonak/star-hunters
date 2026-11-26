import assert from 'node:assert/strict';
import test from 'node:test';

import { upsertRecentPatch } from '../src/hooks/useRecentPatches.ts';
import { ClientPatchCache, livePatchCacheKey } from '../src/services/clientPatchCache.ts';
import { loadLiveCoordinatesFlow } from '../src/services/progressivePatchLoader.ts';
import { NetworkQualityTracker, readNavigatorConnectionHint } from '../src/utils/networkQuality.ts';
import { stepAdjacentPatchCoordinates } from '../src/utils/wcsViewport.ts';
import { readPresetManifest } from './helpers/fixtures.mjs';

test('NetworkQualityTracker: < 3 Mbps drops to float16 and >= 5 Mbps recovers to float32', () => {
  const tracker = new NetworkQualityTracker(() => ({ isSlow: false, downlinkMbps: null }));
  assert.equal(tracker.resolveTargetDtype('auto'), 'float32');
  assert.equal(tracker.resolveTargetDtype('compressed'), 'float16');

  // Simulate 1 MB in 4000 ms (~2.0 Mbps < 3.0 Mbps)
  const slowSnap = tracker.recordTransfer(1_000_000, 4000);
  assert.equal(slowSnap.isSlow, true);
  assert.equal(tracker.resolveTargetDtype('auto'), 'float16');
  assert.equal(tracker.resolveTargetDtype('full'), 'float32');

  // Simulate 4.0 Mbps (between 3 and 5 Mbps) -> hysteresis keeps slow=true
  tracker.recordTransfer(1_000_000, 2000);
  assert.equal(tracker.resolveTargetDtype('auto'), 'float16');

  // Simulate repeated fast transfers (>= 5 Mbps) -> recovers to float32
  for (let i = 0; i < 6; i += 1) {
    tracker.recordTransfer(1_000_000, 500); // 16 Mbps
  }
  assert.equal(tracker.getSnapshot().isSlow, false);
  assert.equal(tracker.resolveTargetDtype('auto'), 'float32');
});

test('readNavigatorConnectionHint detects slow 2g/3g and saveData hints', () => {
  assert.equal(readNavigatorConnectionHint({ effectiveType: '3g', downlink: 1.5 }).isSlow, true);
  assert.equal(readNavigatorConnectionHint({ saveData: true, downlink: 20 }).isSlow, true);
  assert.equal(readNavigatorConnectionHint({ effectiveType: '4g', downlink: 15 }).isSlow, false);
});

test('loadLiveCoordinatesFlow: auto mode falls back from stalled float32 to float16 and separates cache entries', async () => {
  const manifest = readPresetManifest('wise_0855');
  const liveManifest = {
    ...manifest,
    epochs: manifest.epochs.slice(0, 1).map((ep, idx) => ({
      ...ep,
      binary_dtype: 'float32',
      detectors_data_url: `/api/v1/frames/test_grid/ep_${idx}_detectors.bin`,
      flags_mask_url: `/api/v1/frames/test_grid/ep_${idx}_flags.bin`,
    })),
    default_spectrum_url: null,
  };

  const float16Frame = new Uint16Array(6 * 256 * 256);
  const float32Frame = new Float32Array(6 * 256 * 256);
  const flagsFrame = new Uint8Array(256 * 256);
  const fetchedUrls = [];
  let stallFloat32 = true;

  const mockFetch = async (url, init) => {
    fetchedUrls.push(url);
    if (url.startsWith('/api/v1/catalogs')) {
      return new Response(
        JSON.stringify({ grid_key: 'test_grid', catalog_overlays: manifest.catalog_overlays }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
    if (url.startsWith('/api/v1/sky-patch')) {
      return new Response(JSON.stringify(liveManifest), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.includes('precision=float32')) {
      if (stallFloat32) {
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            const err = new Error('Aborted');
            err.name = 'AbortError';
            reject(err);
          });
        });
      }
      return new Response(float32Frame.buffer, {
        status: 200,
        headers: { 'X-Binary-Dtype': 'float32' },
      });
    }
    if (url.includes('precision=float16')) {
      return new Response(float16Frame.buffer, {
        status: 200,
        headers: { 'X-Binary-Dtype': 'float16' },
      });
    }
    if (url.includes('flags.bin')) {
      return new Response(flagsFrame.buffer, { status: 200 });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const cache = new ClientPatchCache(10);
  const tracker = new NetworkQualityTracker(() => ({ isSlow: false, downlinkMbps: null }));
  let firstReady = null;
  await loadLiveCoordinatesFlow(133.8, -7.25, 5.0, {
    cache,
    fetchImpl: mockFetch,
    qualityMode: 'auto',
    networkTracker: tracker,
    stallTimeoutMs: 35,
    onReady: (res) => {
      firstReady = res;
    },
  });

  assert.ok(firstReady);
  assert.equal(firstReady.loadedDtype, 'float16');
  assert.equal(firstReady.degradedOnAutoFallback, true);
  assert.ok(fetchedUrls.some((u) => u.includes('precision=float32')));
  assert.ok(fetchedUrls.some((u) => u.includes('precision=float16')));
  assert.equal(
    await cache.hasPatch(livePatchCacheKey('ra133.80_dec-7.25_fov5.0', 'float16')),
    true,
  );
  assert.equal(
    await cache.hasPatch(livePatchCacheKey('ra133.80_dec-7.25_fov5.0', 'float32')),
    false,
  );

  // Now un-stall float32 and refetch in 'full' mode -> upgrades cache with distinct float32 entry
  stallFloat32 = false;
  let fullReady = null;
  await loadLiveCoordinatesFlow(133.8, -7.25, 5.0, {
    cache,
    fetchImpl: mockFetch,
    qualityMode: 'full',
    networkTracker: tracker,
    bypassClientCache: true,
    onReady: (res) => {
      fullReady = res;
    },
  });
  assert.ok(fullReady);
  assert.equal(fullReady.loadedDtype, 'float32');
  assert.equal(
    await cache.hasPatch(livePatchCacheKey('ra133.80_dec-7.25_fov5.0', 'float32')),
    true,
  );
});

test('ClientPatchCache expires entries older than TTL', async () => {
  const cache = new ClientPatchCache(10, 1000);
  const manifest = readPresetManifest('wise_0855');
  const oldTime = Date.now() - 5000;
  await cache.putPatch({
    cacheKey: 'expired_key',
    dtype: 'float32',
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    spectrum: null,
    storedAt: oldTime,
    lastAccessed: oldTime,
  });
  assert.equal(await cache.getPatch('expired_key'), null);
});

test('stepAdjacentPatchCoordinates steps 0.05 deg N/S/E/W with RA wraparound and upsertRecentPatch caps at 12', () => {
  const north = stepAdjacentPatchCoordinates(0.0, 0.0, 'north');
  assert.deepEqual(north, { raQ: 0, decQ: 0.05 });

  const westWrap = stepAdjacentPatchCoordinates(0.02, 0.0, 'west');
  assert.equal(westWrap.raQ, 359.95);
  assert.equal(westWrap.decQ, 0);

  let recents = [];
  for (let i = 0; i < 15; i += 1) {
    recents = upsertRecentPatch(recents, {
      id: `id_${i}`,
      kind: 'live',
      ra: i,
      dec: 0,
      fovArcmin: 5,
      title: `Patch ${i}`,
      coordsLabel: `${i}, 0`,
      openedAt: i,
    });
  }
  assert.equal(recents.length, 12);
  assert.equal(recents[0].id, 'id_14');
});
