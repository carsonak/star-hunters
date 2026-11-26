import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ClientPatchCache,
  gridCacheKey,
  presetCacheKey,
} from '../src/services/clientPatchCache.ts';
import {
  applyContrastStretch,
  computeLocalPixelReadout,
  renderEpochBuffersToRgba,
} from '../src/utils/pixelEngine.ts';
import { computeDetectorWeights, computeGridKey } from '../src/utils/wcsViewport.ts';
import { makeGaussianEpochPlanes, readPresetManifest } from './helpers/fixtures.mjs';

test('Wavelength Dim & Pop detector weights isolate cold IR detectors (3.8 - 5.0 um)', () => {
  const coldWeights = computeDetectorWeights([3.8, 5.0]);
  // Detectors 1..4 (indices 0..3) are at the 5% floor, Detectors 5..6 (indices 4..5) are at 100%
  assert.ok(Math.abs(coldWeights[0] - 0.05) < 1e-6);
  assert.ok(Math.abs(coldWeights[2] - 0.05) < 1e-6);
  assert.ok(Math.abs(coldWeights[4] - 1.0) < 1e-6);
  assert.ok(Math.abs(coldWeights[5] - 1.0) < 1e-6);

  // SkyView optical/NIR range [0.65, 0.85] preserves Detectors 1–2 at full brightness
  const skyViewFullWeights = computeDetectorWeights([0.65, 0.85]);
  assert.ok(Math.abs(skyViewFullWeights[0] - 1.0) < 1e-6);
  assert.ok(Math.abs(skyViewFullWeights[1] - 1.0) < 1e-6);
  assert.ok(Math.abs(skyViewFullWeights[2] - 0.05) < 1e-6);

  // Sub-band SkyView slice [0.65, 0.75] scales Detectors 1–2 smoothly without blacking out
  const skyViewRedWeights = computeDetectorWeights([0.65, 0.75]);
  assert.ok(skyViewRedWeights[0] > 0.5 && skyViewRedWeights[0] < 0.55);
});

test('ISS-11: subtracting an epoch from itself produces a uniform neutral background', () => {
  const epoch = makeGaussianEpochPlanes([{ x: 72, y: 88, peak: 9.0, sigma: 1.7 }]);
  const rgba = renderEpochBuffersToRgba({
    epochBuffers: [epoch],
    comparisonMode: 'subtract',
    passAIndex: 0,
    passBIndex: 0,
    contrastStretch: 'asinh',
    wavelengthRangeUm: [0.75, 5.0],
  });

  let nonNeutralPixels = 0;
  for (let i = 0; i < 256 * 256; i += 1) {
    const outIdx = i * 4;
    if (
      rgba[outIdx] !== 0 ||
      rgba[outIdx + 1] !== 0 ||
      rgba[outIdx + 2] !== 0 ||
      rgba[outIdx + 3] !== 255
    ) {
      nonNeutralPixels += 1;
    }
  }
  assert.equal(nonNeutralPixels, 0);
});

test('ISS-11: cold wavelength window dims D1-D3 by >=90% and preserves D5-D6 at full intensity', () => {
  const epoch = Array.from({ length: 6 }, () => {
    const plane = new Float32Array(256 * 256);
    plane.fill(0.01);
    return plane;
  });
  const addSpot = (plane, x, y, peak, sigma = 1.5) => {
    const twoSigmaSq = 2 * sigma * sigma;
    for (let py = Math.max(0, y - 6); py <= Math.min(255, y + 6); py += 1) {
      for (let px = Math.max(0, x - 6); px <= Math.min(255, x + 6); px += 1) {
        const r2 = (px - x) * (px - x) + (py - y) * (py - y);
        plane[py * 256 + px] += peak * Math.exp(-r2 / twoSigmaSq);
      }
    }
  };

  // Hot short-wavelength star: detectors D1-D3 (indices 0..2).
  addSpot(epoch[0], 64, 64, 8.0);
  addSpot(epoch[1], 64, 64, 8.0);
  addSpot(epoch[2], 64, 64, 8.0);
  // Cold brown dwarf: detectors D5-D6 (indices 4..5).
  addSpot(epoch[4], 192, 192, 8.0);
  addSpot(epoch[5], 192, 192, 8.0);

  const renderAt = (wavelengthRangeUm, contrastStretch) =>
    renderEpochBuffersToRgba({
      epochBuffers: [epoch],
      activeEpochIndex: 0,
      comparisonMode: 'blink',
      contrastStretch,
      wavelengthRangeUm,
    });

  const hotIndex = (64 * 256 + 64) * 4;
  const coldIndex = (192 * 256 + 192) * 4;

  for (const contrastStretch of ['asinh', 'linear', 'zscale']) {
    const fullRange = renderAt([0.75, 5.0], contrastStretch);
    const coldRange = renderAt([3.8, 5.0], contrastStretch);

    // Blue channel of the hot star must dim by at least 90%.
    assert.ok(
      coldRange[hotIndex + 2] <= fullRange[hotIndex + 2] * 0.1,
      `Expected hot-star blue ${coldRange[hotIndex + 2]} <= 10% of ${fullRange[hotIndex + 2]} (${contrastStretch})`,
    );
    // Green channel (D3) dims as well.
    assert.ok(coldRange[hotIndex + 1] <= fullRange[hotIndex + 1] * 0.1);

    // The cold brown dwarf's red channel is untouched by the wavelength window.
    assert.equal(coldRange[coldIndex], fullRange[coldIndex]);
    assert.ok(coldRange[coldIndex] > 240);
  }
});

test('ISS-11: blink, subtract and coadd combine cached epochs and flag bits tint the frame', () => {
  const passA = makeGaussianEpochPlanes([{ x: 60, y: 60, peak: 7.0 }], 0.005);
  const passB = makeGaussianEpochPlanes([{ x: 180, y: 180, peak: 7.0 }], 0.005);

  const blinkA = renderEpochBuffersToRgba({
    epochBuffers: [passA, passB],
    activeEpochIndex: 0,
    comparisonMode: 'blink',
    wavelengthRangeUm: [0.75, 5.0],
  });
  const blinkB = renderEpochBuffersToRgba({
    epochBuffers: [passA, passB],
    activeEpochIndex: 1,
    comparisonMode: 'blink',
    wavelengthRangeUm: [0.75, 5.0],
  });
  assert.ok(blinkA[(60 * 256 + 60) * 4] > 0);
  assert.equal(blinkB[(60 * 256 + 60) * 4], 0);

  // Pass B - Pass A: the source at (180,180) is a positive (warm) residual,
  // the source at (60,60) is a negative (cool) residual.
  const positive = renderEpochBuffersToRgba({
    epochBuffers: [passA, passB],
    comparisonMode: 'subtract',
    passAIndex: 0,
    passBIndex: 1,
    wavelengthRangeUm: [0.75, 5.0],
  });
  const positiveIdx = (180 * 256 + 180) * 4;
  const negativeIdx = (60 * 256 + 60) * 4;
  assert.ok(positive[positiveIdx] > positive[positiveIdx + 2]);
  // The same Pass B - Pass A frame also exposes the source that disappeared
  // from (60,60) as a cool cyan/blue residual.
  assert.ok(positive[negativeIdx + 2] > positive[negativeIdx]);

  // Co-add averages both passes, so both sources remain visible.
  const coadd = renderEpochBuffersToRgba({
    epochBuffers: [passA, passB],
    comparisonMode: 'coadd',
    coaddIndices: [0, 1],
    wavelengthRangeUm: [0.75, 5.0],
  });
  assert.ok(coadd[(60 * 256 + 60) * 4] > 0);
  assert.ok(coadd[(180 * 256 + 180) * 4] > 0);

  // Optional flags overlay tints masked pixels magenta/amber.
  const flags = new Uint8Array(256 * 256);
  flags[60 * 256 + 60] = 0x01;
  flags[60 * 256 + 61] = 0x02;
  const noFlags = renderEpochBuffersToRgba({
    epochBuffers: [passA],
    activeEpochIndex: 0,
    comparisonMode: 'blink',
    wavelengthRangeUm: [0.75, 5.0],
    flagsBuffers: [flags],
    showFlagsOverlay: false,
  });
  const withFlags = renderEpochBuffersToRgba({
    epochBuffers: [passA],
    activeEpochIndex: 0,
    comparisonMode: 'blink',
    wavelengthRangeUm: [0.75, 5.0],
    flagsBuffers: [flags],
    showFlagsOverlay: true,
  });
  assert.notDeepEqual(
    Array.from(withFlags.slice(60 * 256 * 4, 62 * 256 * 4)),
    Array.from(noFlags.slice(60 * 256 * 4, 62 * 256 * 4)),
  );
});

test('ISS-11: rendering a 256x256 RGBA frame completes in <5 ms', () => {
  const passA = makeGaussianEpochPlanes([
    { x: 64, y: 64, peak: 8.0 },
    { x: 190, y: 120, peak: 5.0 },
  ]);
  const passB = makeGaussianEpochPlanes([
    { x: 74, y: 70, peak: 8.0 },
    { x: 190, y: 120, peak: 5.0 },
  ]);
  const params = {
    epochBuffers: [passA, passB],
    comparisonMode: 'coadd',
    coaddIndices: [0, 1],
    contrastStretch: 'asinh',
    wavelengthRangeUm: [0.75, 5.0],
  };

  // Warm the JIT so the measurement reflects steady-state per-frame cost.
  for (let i = 0; i < 30; i += 1) {
    renderEpochBuffersToRgba(params);
  }
  const iterations = 50;
  const startedAt = performance.now();
  let last;
  for (let i = 0; i < iterations; i += 1) {
    last = renderEpochBuffersToRgba(params);
  }
  const averageMs = (performance.now() - startedAt) / iterations;
  assert.equal(last.length, 256 * 256 * 4);
  assert.ok(averageMs < 5.0, `Expected <5 ms per frame, measured ${averageMs.toFixed(2)} ms`);
});

test('ISS-11: contrast stretches are monotonic and clamped to [0, 1]', () => {
  for (const stretch of ['asinh', 'linear', 'zscale']) {
    assert.equal(applyContrastStretch(-3, stretch), 0);
    assert.equal(applyContrastStretch(1e6, stretch), 1);
    let previous = -1;
    for (let value = 0; value <= 12; value += 0.25) {
      const stretched = applyContrastStretch(value, stretch);
      assert.ok(stretched >= 0 && stretched <= 1);
      assert.ok(stretched >= previous);
      previous = stretched;
    }
  }
});

test('Phase 1 pixelEngine: coadd ORs flagsBuffers across coaddIndices and reusable outBuffer matches fresh allocation', () => {
  const epoch0 = makeGaussianEpochPlanes([{ x: 100, y: 100, peak: 5.0 }], 0.01);
  const epoch1 = makeGaussianEpochPlanes([{ x: 140, y: 140, peak: 5.0 }], 0.01);
  const flags0 = new Uint8Array(256 * 256);
  const flags1 = new Uint8Array(256 * 256);
  const p0 = 10 * 256 + 10;
  const p1 = 20 * 256 + 20;
  flags0[p0] = 0x01; // amber flag only in epoch 0
  flags1[p1] = 0x02; // magenta flag only in epoch 1

  const coaddNoFlags = renderEpochBuffersToRgba({
    epochBuffers: [epoch0, epoch1],
    flagsBuffers: [flags0, flags1],
    comparisonMode: 'coadd',
    coaddIndices: [0, 1],
    showFlagsOverlay: false,
  });
  const coaddWithFlags = renderEpochBuffersToRgba({
    epochBuffers: [epoch0, epoch1],
    flagsBuffers: [flags0, flags1],
    comparisonMode: 'coadd',
    coaddIndices: [0, 1],
    showFlagsOverlay: true,
  });

  // Both epoch 0's pixel (10,10) and epoch 1's pixel (20,20) must be tinted
  assert.notDeepEqual(
    Array.from(coaddWithFlags.slice(p0 * 4, p0 * 4 + 4)),
    Array.from(coaddNoFlags.slice(p0 * 4, p0 * 4 + 4)),
  );
  assert.notDeepEqual(
    Array.from(coaddWithFlags.slice(p1 * 4, p1 * 4 + 4)),
    Array.from(coaddNoFlags.slice(p1 * 4, p1 * 4 + 4)),
  );
  // Unflagged pixel remains untouched
  const cleanIdx = (30 * 256 + 30) * 4;
  assert.deepEqual(
    Array.from(coaddWithFlags.slice(cleanIdx, cleanIdx + 4)),
    Array.from(coaddNoFlags.slice(cleanIdx, cleanIdx + 4)),
  );

  // Subset coaddIndices: [1] should only tint p1, not p0
  const coaddOnlySecond = renderEpochBuffersToRgba({
    epochBuffers: [epoch0, epoch1],
    flagsBuffers: [flags0, flags1],
    comparisonMode: 'coadd',
    coaddIndices: [1],
    showFlagsOverlay: true,
  });
  const coaddOnlySecondNoFlags = renderEpochBuffersToRgba({
    epochBuffers: [epoch0, epoch1],
    flagsBuffers: [flags0, flags1],
    comparisonMode: 'coadd',
    coaddIndices: [1],
    showFlagsOverlay: false,
  });
  assert.deepEqual(
    Array.from(coaddOnlySecond.slice(p0 * 4, p0 * 4 + 4)),
    Array.from(coaddOnlySecondNoFlags.slice(p0 * 4, p0 * 4 + 4)),
  );
  assert.notDeepEqual(
    Array.from(coaddOnlySecond.slice(p1 * 4, p1 * 4 + 4)),
    Array.from(coaddOnlySecondNoFlags.slice(p1 * 4, p1 * 4 + 4)),
  );

  // Reusable outBuffer equivalence across blink, subtract, coadd, and empty buffers
  const reusable = new Uint8ClampedArray(256 * 256 * 4);
  for (const comparisonMode of ['blink', 'subtract', 'coadd']) {
    const fresh = renderEpochBuffersToRgba({
      epochBuffers: [epoch0, epoch1],
      flagsBuffers: [flags0, flags1],
      comparisonMode,
      activeEpochIndex: 0,
      passAIndex: 0,
      passBIndex: 1,
      coaddIndices: [0, 1],
      showFlagsOverlay: true,
    });
    const inPlace = renderEpochBuffersToRgba({
      epochBuffers: [epoch0, epoch1],
      flagsBuffers: [flags0, flags1],
      comparisonMode,
      activeEpochIndex: 0,
      passAIndex: 0,
      passBIndex: 1,
      coaddIndices: [0, 1],
      showFlagsOverlay: true,
      outBuffer: reusable,
    });
    assert.equal(inPlace, reusable);
    assert.deepEqual(inPlace, fresh);
  }

  // Non-zero byteOffset sub-view works identically and invalid size throws RangeError
  const padded = new Uint8ClampedArray(256 * 256 * 4 + 16);
  const offsetView = padded.subarray(4, 4 + 256 * 256 * 4);
  const freshBlink = renderEpochBuffersToRgba({ epochBuffers: [epoch0], comparisonMode: 'blink' });
  const viewResult = renderEpochBuffersToRgba({
    epochBuffers: [epoch0],
    comparisonMode: 'blink',
    outBuffer: offsetView,
  });
  assert.equal(viewResult, offsetView);
  assert.deepEqual(viewResult, freshBlink);
  assert.throws(
    () =>
      renderEpochBuffersToRgba({ epochBuffers: [epoch0], outBuffer: new Uint8ClampedArray(16) }),
    RangeError,
  );
});

test('Phase 1 computeLocalPixelReadout computes exact running variance and clamps boundaries', () => {
  const uniformPlane = new Float32Array(256 * 256).fill(4.25);
  const uniformEpoch = Array.from({ length: 6 }, () => uniformPlane);
  const cornerReadout = computeLocalPixelReadout({
    epochBuffers: [uniformEpoch],
    px: -5,
    py: 300,
  });
  assert.equal(cornerReadout.x, 0);
  assert.equal(cornerReadout.y, 255);
  assert.ok(Math.abs(cornerReadout.flux - 4.25) < 1e-5);
  assert.equal(cornerReadout.variance, 0);

  const spikePlane = new Float32Array(256 * 256).fill(10);
  spikePlane[128 * 256 + 128] = 46;
  const spikeEpoch = Array.from({ length: 6 }, () => spikePlane);
  const spikeReadout = computeLocalPixelReadout({
    epochBuffers: [spikeEpoch],
    px: 128.9,
    py: 128.1,
  });
  assert.equal(spikeReadout.x, 128);
  assert.equal(spikeReadout.y, 128);
  assert.ok(Math.abs(spikeReadout.flux - 46) < 1e-5);
  // Exact 3x3 population variance: (8*10^2 + 46^2)/9 - 14^2 = 324 - 196 = 128
  assert.ok(Math.abs(spikeReadout.variance - 128) < 1e-4);
});

test('ClientPatchCache caches sky patches in memory and evicts oldest entry on LRU overflow', async () => {
  const cache = new ClientPatchCache(2);
  const manifest = readPresetManifest('wise_0855');

  await cache.putPatch({
    cacheKey: 'k1',
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    storedAt: 1,
    lastAccessed: 1,
  });
  await cache.putPatch({
    cacheKey: 'k2',
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    storedAt: 2,
    lastAccessed: 2,
  });
  // Access k1 so k2 becomes the oldest
  const hit1 = await cache.getPatch('k1');
  assert.ok(hit1 !== null);

  await cache.putPatch({
    cacheKey: 'k3',
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    storedAt: 3,
    lastAccessed: 3,
  });

  assert.ok((await cache.getPatch('k1')) !== null);
  assert.equal(await cache.getPatch('k2'), null);
  assert.ok((await cache.getPatch('k3')) !== null);
});

test('ISS-10: cached grid_key and preset:{slug} are served from memory in <5 ms with zero network requests', async () => {
  const cache = new ClientPatchCache(4);
  const manifest = readPresetManifest('wise_0855');
  const gridKey = computeGridKey(133.7951, -7.2451, 5.0);
  const presetKey = presetCacheKey('wise_0855');

  assert.equal(gridCacheKey(gridKey), 'ra133.80_dec-7.25_fov5.0');
  assert.equal(presetKey, 'preset:wise_0855');
  assert.equal(gridCacheKey('  ra133.80_dec-7.25_fov5.0  '), gridKey);
  assert.equal(presetCacheKey('  wise_0855  '), presetKey);

  const makeBundle = (cacheKey) => ({
    cacheKey,
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    storedAt: Date.now(),
    lastAccessed: Date.now(),
  });

  const originalFetch = globalThis.fetch;
  let networkCalls = 0;
  globalThis.fetch = (...args) => {
    networkCalls += 1;
    return Promise.reject(new Error(`Unexpected network request: ${String(args[0])}`));
  };

  try {
    await cache.putPatch(makeBundle(gridKey));
    await cache.putPatch(makeBundle(presetKey));
    await cache.putCatalogs(gridKey, {
      query: manifest.query,
      wcs: manifest.wcs,
      catalog_overlays: manifest.catalog_overlays,
    });

    // A loader must not run for keys that are already cached.
    const gridHit = await cache.getOrFetchPatch(gridKey, async () => {
      networkCalls += 1;
      return null;
    });
    const presetHit = await cache.getOrFetchPatch(presetKey, async () => {
      networkCalls += 1;
      return null;
    });
    const catalogHit = await cache.getOrFetchCatalogs(gridKey, async () => {
      networkCalls += 1;
      return null;
    });
    assert.ok(gridHit);
    assert.ok(presetHit);
    assert.ok(catalogHit);

    // Warm the JIT then measure the average hot-path retrieval time.
    for (let i = 0; i < 10; i += 1) {
      await cache.getPatch(gridKey);
      await cache.getPatch(presetKey);
    }
    const iterations = 200;
    const startedAt = performance.now();
    for (let i = 0; i < iterations; i += 1) {
      const hit = await cache.getPatch(i % 2 === 0 ? gridKey : presetKey);
      assert.ok(hit);
    }
    const averageMs = (performance.now() - startedAt) / iterations;
    assert.ok(averageMs < 5, `cached patch retrieval averaged ${averageMs.toFixed(3)} ms`);

    assert.ok(cache.getPatchSync(gridKey));
    assert.ok(cache.getPatchSync(presetKey));
    assert.equal(networkCalls, 0, 'cached retrieval must never touch the network');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('ISS-10: ClientPatchCache LRU-evicts catalogs and works without IndexedDB in headless environments', async () => {
  const cache = new ClientPatchCache(2);
  const manifest = readPresetManifest('wise_0855');
  const catalogsFor = (ra) => ({
    query: { ...manifest.query, ra },
    wcs: manifest.wcs,
    catalog_overlays: manifest.catalog_overlays,
  });

  // Node has no indexedDB: every operation must gracefully fall back to memory.
  assert.equal(cache.isPersistentStorageAvailable(), false);
  assert.equal(cache.capacity, 2);

  await cache.putCatalogs('grid_a', catalogsFor(10));
  await cache.putCatalogs('grid_b', catalogsFor(20));
  assert.equal((await cache.getCatalogs('grid_a')).query.ra, 10);

  await cache.putCatalogs('grid_c', catalogsFor(30));
  assert.equal(await cache.getCatalogs('grid_b'), null, 'LRU catalog grid_b should be evicted');
  assert.equal((await cache.getCatalogs('grid_a')).query.ra, 10);
  assert.equal((await cache.getCatalogs('grid_c')).query.ra, 30);
  assert.equal(await cache.hasCatalogs('grid_b'), false);
  assert.equal(await cache.hasCatalogs('grid_c'), true);

  assert.deepEqual(cache.size, { patches: 0, catalogs: 2 });
  const stats = cache.getStats();
  assert.equal(stats.maxEntries, 2);
  assert.equal(stats.memoryCatalogs, 2);
  assert.ok(stats.catalogHits >= 2);
  assert.ok(stats.catalogMisses >= 1);

  await cache.deleteCatalogs('grid_a');
  assert.equal(await cache.getCatalogs('grid_a'), null);

  const patch = {
    cacheKey: 'grid_patch',
    manifest,
    epochBuffers: [],
    flagsBuffers: [],
    storedAt: Date.now(),
    lastAccessed: Date.now(),
  };
  await cache.putPatch(patch);
  assert.equal(await cache.hasPatch('grid_patch'), true);
  assert.ok(await cache.getPatch('grid_patch'));
  await cache.deletePatch('grid_patch');
  assert.equal(await cache.hasPatch('grid_patch'), false);

  await cache.putCatalogs('grid_d', catalogsFor(40));
  await cache.clearAll();
  assert.deepEqual(cache.size, { patches: 0, catalogs: 0 });
});

test('Phase 1 TieredLruStore persists to IndexedDB, restores after clearMemory, and prunes LRU entries', async () => {
  const stores = new Map();
  const makeRequest = (executor) => {
    const req = { result: undefined, onsuccess: null, onerror: null };
    queueMicrotask(() => {
      try {
        req.result = executor();
        req.onsuccess?.({ target: req });
      } catch {
        req.onerror?.({ target: req });
      }
    });
    return req;
  };
  const fakeDb = {
    objectStoreNames: { contains: (name) => stores.has(name) },
    createObjectStore: (name) => {
      if (!stores.has(name)) stores.set(name, new Map());
    },
    transaction: (names) => {
      const storeList = Array.isArray(names) ? names : [names];
      const tx = { oncomplete: null, onerror: null, onabort: null };
      let pending = 0;
      const track = (fn) => {
        pending += 1;
        return makeRequest(() => {
          const res = fn();
          queueMicrotask(() => {
            pending -= 1;
            if (pending === 0) tx.oncomplete?.({ target: tx });
          });
          return res;
        });
      };
      tx.objectStore = (name) => {
        if (!storeList.includes(name) || !stores.has(name)) {
          throw new Error(`Unknown store ${name}`);
        }
        const map = stores.get(name);
        return {
          get: (key) => track(() => map.get(key)),
          put: (value, key) => track(() => map.set(key, value)),
          delete: (key) => track(() => map.delete(key)),
          clear: () => track(() => map.clear()),
          openCursor: () => {
            const entries = Array.from(map.entries());
            let idx = 0;
            const req = { result: null, onsuccess: null, onerror: null };
            const advance = () => {
              queueMicrotask(() => {
                if (idx < entries.length) {
                  const [key, value] = entries[idx];
                  idx += 1;
                  req.result = { key, value, continue: advance };
                  req.onsuccess?.({ target: req });
                } else {
                  req.result = null;
                  req.onsuccess?.({ target: req });
                  queueMicrotask(() => {
                    if (pending === 0) tx.oncomplete?.({ target: tx });
                  });
                }
              });
            };
            advance();
            return req;
          },
        };
      };
      return tx;
    },
  };

  const origDesc = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
  Object.defineProperty(globalThis, 'indexedDB', {
    configurable: true,
    value: {
      open: () => {
        const req = {
          result: fakeDb,
          onupgradeneeded: null,
          onsuccess: null,
          onerror: null,
          onblocked: null,
        };
        queueMicrotask(() => {
          req.onupgradeneeded?.({ target: req });
          req.onsuccess?.({ target: req });
        });
        return req;
      },
    },
  });

  try {
    const cache = new ClientPatchCache(2);
    assert.equal(cache.isPersistentStorageAvailable(), true);

    const mkBundle = (key) => ({
      cacheKey: key,
      manifest: {
        query: { ra: 0, dec: 0, fov_arcmin: 5, grid_key: key },
        epochs: [],
        catalog_overlays: { known_objects: [] },
      },
      epochBuffers: [],
      flagsBuffers: [],
      spectrum: null,
      storedAt: Date.now(),
      lastAccessed: Date.now(),
    });

    await cache.putPatch(mkBundle('k1'));
    await cache.putPatch(mkBundle('k2'));
    // Touch k1 in memory so k2 becomes the least recently used entry
    assert.ok(await cache.getPatch('k1'));
    // Inserting k3 must evict k2 from both Tier 1 memory and Tier 2 IndexedDB
    await cache.putPatch(mkBundle('k3'));

    cache.clearMemory();
    assert.equal(cache.size.patches, 0);
    assert.equal(await cache.hasPatch('k2'), false);
    assert.ok(await cache.getPatch('k1'), 'k1 must restore from IndexedDB after clearMemory');
    assert.ok(await cache.getPatch('k3'), 'k3 must restore from IndexedDB after clearMemory');
    assert.equal(cache.size.patches, 2);

    await cache.clearAll();
    assert.equal(await cache.hasPatch('k1'), false);
  } finally {
    if (origDesc) {
      Object.defineProperty(globalThis, 'indexedDB', origDesc);
    } else {
      delete globalThis.indexedDB;
    }
  }
});
