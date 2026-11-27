import assert from 'node:assert/strict';
import test from 'node:test';

import { prewarmBenchmarkTargets, waitForIdleTurn } from '../src/hooks/useObservatoryWarmup.ts';
import { ClientPatchCache } from '../src/services/clientPatchCache.ts';
import {
  loadLiveCoordinatesFlow,
  loadPresetPatchFlow,
} from '../src/services/progressivePatchLoader.ts';
import { detectAndTrackBodiesAtClick } from '../src/utils/clickTracker.ts';
import {
  clearCheckedObjectIds,
  selectAllObjectIds,
  toggleObjectId,
} from '../src/utils/knownObjectSelection.ts';
import { renderEpochBuffersToRgba } from '../src/utils/pixelEngine.ts';
import {
  URL_STATE_DEFAULTS,
  parseUrlState,
  replaceUrlState,
  serializeUrlState,
} from '../src/utils/urlStateSync.ts';
import { clampViewportPanZoom } from '../src/utils/wcsViewport.ts';
import { loadPresetEpochAndFlags, readPresetManifest } from './helpers/fixtures.mjs';

test('ISS-14: parseUrlState restores exact coordinates, comparison mode, and wavelength filter', () => {
  const parsed = parseUrlState(
    '?target=barnards_star&ra=269.4521&dec=4.6934&fov=5&mode=subtract&wlMin=3.8&wlMax=5.0',
  );

  assert.equal(parsed.target, 'barnards_star');
  assert.equal(parsed.ra, 269.4521);
  assert.equal(parsed.dec, 4.6934);
  assert.equal(parsed.fov, 5);
  assert.equal(parsed.mode, 'subtract');
  assert.equal(parsed.wlMin, 3.8);
  assert.equal(parsed.wlMax, 5.0);

  // Acceptance URL without a preset still resolves the exact sky state.
  const acceptance = parseUrlState('?ra=269.4521&dec=4.6934&mode=subtract&wlMin=3.8&wlMax=5.0');
  assert.equal(acceptance.target, null);
  assert.equal(acceptance.ra, 269.4521);
  assert.equal(acceptance.dec, 4.6934);
  assert.equal(acceptance.mode, 'subtract');
  assert.deepEqual([acceptance.wlMin, acceptance.wlMax], [3.8, 5.0]);
});

test('ISS-14: parseUrlState applies scientific defaults and ignores invalid ranges', () => {
  const empty = parseUrlState('');
  assert.equal(empty.target, null);
  assert.equal(empty.ra, null);
  assert.equal(empty.dec, null);
  assert.equal(empty.fov, URL_STATE_DEFAULTS.fov);
  assert.equal(empty.mode, URL_STATE_DEFAULTS.mode);
  assert.equal(empty.wlMin, URL_STATE_DEFAULTS.wlMin);
  assert.equal(empty.wlMax, URL_STATE_DEFAULTS.wlMax);

  const invalid = parseUrlState(
    '?ra=999&dec=-120&fov=0&mode=bogus&wlMin=abc&wlMax=9&target=%3Cscript%3E',
  );
  assert.equal(invalid.target, null);
  assert.equal(invalid.ra, null);
  assert.equal(invalid.dec, null);
  assert.equal(invalid.fov, URL_STATE_DEFAULTS.fov);
  assert.equal(invalid.mode, URL_STATE_DEFAULTS.mode);
  assert.equal(invalid.wlMin, URL_STATE_DEFAULTS.wlMin);
  assert.equal(invalid.wlMax, URL_STATE_DEFAULTS.wlMax);

  // Reversed wavelength windows are normalized rather than dropped.
  const reversed = parseUrlState('?wlMin=5&wlMax=3.8');
  assert.deepEqual([reversed.wlMin, reversed.wlMax], [3.8, 5]);

  // RA 360 is the same as 0h, and full URLs are accepted.
  const fullUrl = parseUrlState('https://example.test/explore?ra=360&dec=-90&mode=coadd');
  assert.equal(fullUrl.ra, 0);
  assert.equal(fullUrl.dec, -90);
  assert.equal(fullUrl.mode, 'coadd');
});

test('ISS-14: serializeUrlState round-trips the sky view and preserves unrelated params', () => {
  const serialized = serializeUrlState({
    ra: 269.4521,
    dec: 4.6934,
    fov: 5,
    mode: 'subtract',
    wlMin: 3.8,
    wlMax: 5.0,
  });

  assert.equal(serialized, 'ra=269.4521&dec=4.6934&fov=5&mode=subtract&wlMin=3.8&wlMax=5');
  assert.deepEqual(parseUrlState(`?${serialized}`), {
    target: null,
    ra: 269.4521,
    dec: 4.6934,
    fov: 5,
    mode: 'subtract',
    wlMin: 3.8,
    wlMax: 5,
  });

  // Non-sky parameters survive, and `null` explicitly clears a known key.
  const updated = serializeUrlState(
    { target: null, mode: 'coadd' },
    '?target=wise_0855&foo=bar&mode=blink',
  );
  assert.equal(updated, 'mode=coadd&foo=bar');
  assert.equal(parseUrlState(`?${updated}`).target, null);
});

test('ISS-14: replaceUrlState mirrors the view into history.replaceState without navigation', () => {
  const calls = [];
  const history = {
    state: { from: 'test' },
    replaceState(data, unused, url) {
      calls.push({ data, unused, url });
    },
    pushState() {
      throw new Error('ISS-14 must not push a new history entry');
    },
  };
  const location = {
    pathname: '/explore',
    search: '?target=wise_0855&utm_source=share',
    hash: '#sky',
  };

  const url = replaceUrlState(
    {
      target: null,
      ra: 269.4521,
      dec: 4.6934,
      fov: 5,
      mode: 'subtract',
      wlMin: 3.8,
      wlMax: 5.0,
    },
    { history, location },
  );

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].data, { from: 'test' });
  assert.equal(calls[0].unused, '');
  assert.equal(
    url,
    '/explore?ra=269.4521&dec=4.6934&fov=5&mode=subtract&wlMin=3.8&wlMax=5&utm_source=share#sky',
  );
  assert.equal(calls[0].url, url);
});

test('ISS-21: Stage A catalogs populate knownObjects before Stage B cutouts complete and use ?precision=float16', async () => {
  const manifest = readPresetManifest('wise_0855');
  const liveManifest = {
    ...manifest,
    epochs: manifest.epochs.map((ep, idx) => ({
      ...ep,
      binary_dtype: 'float16',
      detectors_data_url: `/api/v1/frames/test_grid/ep_${idx}_detectors.bin`,
      flags_mask_url: `/api/v1/frames/test_grid/ep_${idx}_flags.bin`,
    })),
    default_spectrum_url: null,
  };

  const float16Frame = new Uint16Array(6 * 256 * 256);
  const flagsFrame = new Uint8Array(256 * 256);
  const fetchedUrls = [];
  const timeline = [];

  let releaseStageB;
  const stageBGate = new Promise((resolve) => {
    releaseStageB = resolve;
  });

  const mockFetch = async (url) => {
    fetchedUrls.push(url);
    if (url.startsWith('/api/v1/catalogs')) {
      return new Response(
        JSON.stringify({
          grid_key: 'test_grid',
          catalog_overlays: manifest.catalog_overlays,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
    if (url.startsWith('/api/v1/sky-patch')) {
      await stageBGate;
      return new Response(JSON.stringify(liveManifest), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.includes('detectors.bin')) {
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
  let readyResult = null;
  const flowPromise = loadLiveCoordinatesFlow(133.8, -7.25, 5.0, {
    cache,
    fetchImpl: mockFetch,
    qualityMode: 'compressed',
    onStageChange: (st) => timeline.push(`stage:${st.stage}`),
    onCatalogsReady: (cat) => {
      timeline.push(`catalogs:${cat.catalog_overlays.known_objects.length}`);
      assert.equal(readyResult, null, 'Stage A must resolve before Stage B finishes');
      releaseStageB();
    },
    onReady: (res) => {
      readyResult = res;
      timeline.push(`ready:cache=${res.fromClientCache}`);
    },
  });

  const outcome = await flowPromise;
  assert.equal(outcome.aborted, false);
  assert.equal(outcome.fromClientCache, false);
  assert.ok(readyResult);
  assert.ok(
    timeline.some((entry) => entry.startsWith('catalogs:')),
    'Expected Stage A callback to fire',
  );
  assert.ok(
    timeline.findIndex((e) => e.startsWith('catalogs:')) <
      timeline.findIndex((e) => e.startsWith('ready:')),
  );
  assert.ok(
    fetchedUrls.some((u) => u.includes('detectors.bin?precision=float16')),
    'Expected live detector frames to request ?precision=float16',
  );

  // Second call for the same coordinates must hit ClientPatchCache with 0 HTTP requests
  const beforeCount = fetchedUrls.length;
  let cachedReady = null;
  const cachedOutcome = await loadLiveCoordinatesFlow(133.8, -7.25, 5.0, {
    cache,
    fetchImpl: mockFetch,
    qualityMode: 'compressed',
    onReady: (res) => {
      cachedReady = res;
    },
  });
  assert.equal(cachedOutcome.fromClientCache, true);
  assert.equal(cachedReady.fromClientCache, true);
  assert.equal(fetchedUrls.length, beforeCount, 'Cache hit must issue 0 HTTP requests');
});

test('ISS-21: Rapid navigation cancels stale AbortController so stale responses never overwrite active patch', async () => {
  const manifest = readPresetManifest('wise_0855');
  const float32Frame = new Float32Array(6 * 256 * 256);
  const flagsFrame = new Uint8Array(256 * 256);

  let releaseStale;
  const staleGate = new Promise((resolve) => {
    releaseStale = resolve;
  });

  const mockFetch = async (url, init) => {
    if (url.includes('wise_0855')) {
      await staleGate;
      if (init?.signal?.aborted) {
        const err = new Error('Aborted');
        err.name = 'AbortError';
        throw err;
      }
    }
    if (url.endsWith('manifest.json')) {
      return new Response(JSON.stringify(manifest), { status: 200 });
    }
    if (url.endsWith('_flags.bin')) {
      return new Response(flagsFrame.buffer, { status: 200 });
    }
    if (url.endsWith('.bin')) {
      return new Response(float32Frame.buffer, {
        status: 200,
        headers: { 'X-Binary-Dtype': 'float32' },
      });
    }
    if (url.endsWith('spectrum.json')) {
      return new Response(JSON.stringify({ flux_jy: [], wavelength_um: [] }), { status: 200 });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const cache = new ClientPatchCache(10);
  const readySlugs = [];

  const ctrl1 = new AbortController();
  const p1 = loadPresetPatchFlow('wise_0855', {
    cache,
    fetchImpl: mockFetch,
    signal: ctrl1.signal,
    onReady: () => readySlugs.push('wise_0855'),
  });

  // Immediately abort first navigation and switch to barnards_star
  ctrl1.abort();
  const ctrl2 = new AbortController();
  const p2 = loadPresetPatchFlow('barnards_star', {
    cache,
    fetchImpl: mockFetch,
    signal: ctrl2.signal,
    onReady: () => readySlugs.push('barnards_star'),
  });

  releaseStale();
  const [out1, out2] = await Promise.all([
    p1.catch((e) => ({ aborted: e.name === 'AbortError' })),
    p2,
  ]);
  assert.equal(out1.aborted, true);
  assert.equal(out2.aborted, false);
  assert.deepEqual(readySlugs, ['barnards_star']);
});

test('ISS-21: Changing render options, pan/zoom, known object selection, and tracking makes 0 HTTP requests', () => {
  const { manifest, epochBuffers, flagsBuffers } = loadPresetEpochAndFlags('wise_0855');

  const origFetch = globalThis.fetch;
  let httpCalls = 0;
  globalThis.fetch = async () => {
    httpCalls += 1;
    throw new Error('Zero HTTP requests allowed during client-side interactions');
  };

  try {
    for (const comparisonMode of ['blink', 'subtract', 'coadd']) {
      renderEpochBuffersToRgba({
        epochBuffers,
        flagsBuffers,
        comparisonMode,
        activeEpochIndex: 0,
        passAIndex: 0,
        passBIndex: 1,
        coaddIndices: [0, 1],
        wavelengthRangeUm: [3.8, 5.0],
        contrastStretch: 'asinh',
        showFlagsOverlay: true,
      });
    }
    clampViewportPanZoom(4.0, 0.25, -0.25);
    selectAllObjectIds(manifest.catalog_overlays.known_objects);
    clearCheckedObjectIds();
    toggleObjectId([], manifest.catalog_overlays.known_objects[0].object_id);
    detectAndTrackBodiesAtClick({
      clickX: 128,
      clickY: 128,
      originEpochIndex: 0,
      epochBuffers,
      manifest,
      wavelengthRangeUm: [0.75, 5.0],
    });
    assert.equal(httpCalls, 0);
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('Phase 1 progressivePatchLoader fetches all epoch binary frames concurrently via Promise.all', async () => {
  const manifest = readPresetManifest('wise_0855');
  const float32Frame = new Float32Array(6 * 256 * 256);
  const flagsFrame = new Uint8Array(256 * 256);

  let inFlightDetectors = 0;
  let peakInFlightDetectors = 0;
  let releaseEpochs;
  const epochGate = new Promise((resolve) => {
    releaseEpochs = resolve;
  });

  const mockFetch = async (url) => {
    if (url.endsWith('manifest.json')) {
      return new Response(JSON.stringify(manifest), { status: 200 });
    }
    if (url.endsWith('_flags.bin')) {
      return new Response(flagsFrame.buffer, { status: 200 });
    }
    if (url.endsWith('.bin')) {
      inFlightDetectors += 1;
      peakInFlightDetectors = Math.max(peakInFlightDetectors, inFlightDetectors);
      if (inFlightDetectors === manifest.epochs.length) {
        releaseEpochs();
      }
      await epochGate;
      inFlightDetectors -= 1;
      return new Response(float32Frame.buffer, {
        status: 200,
        headers: { 'X-Binary-Dtype': 'float32' },
      });
    }
    if (url.endsWith('spectrum.json')) {
      return new Response(JSON.stringify({ ra: 0, dec: 0, channels: [] }), { status: 200 });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const cache = new ClientPatchCache(4);
  let loaded = null;
  const outcome = await loadPresetPatchFlow('wise_0855', {
    cache,
    fetchImpl: mockFetch,
    onReady: (res) => {
      loaded = res;
    },
  });

  assert.equal(outcome.aborted, false);
  assert.equal(peakInFlightDetectors, manifest.epochs.length);
  assert.equal(loaded.epochBuffers.length, manifest.epochs.length);
  assert.equal(loaded.flagsBuffers.length, manifest.epochs.length);
});

test('Phase 1 progressivePatchLoader treats aborted non-ok frame responses as AbortError', async () => {
  const manifest = readPresetManifest('wise_0855');
  const controller = new AbortController();
  const mockFetch = async (url) => {
    if (url.endsWith('manifest.json')) {
      return new Response(JSON.stringify(manifest), { status: 200 });
    }
    controller.abort();
    return new Response(null, { status: 500 });
  };

  const cache = new ClientPatchCache(2);
  await assert.rejects(
    () =>
      loadPresetPatchFlow('wise_0855', {
        cache,
        fetchImpl: mockFetch,
        signal: controller.signal,
        onReady: () => {
          throw new Error('onReady must not run after abort');
        },
      }),
    (err) => err instanceof Error && err.name === 'AbortError',
  );
});

test('Phase 3 useObservatoryWarmup defers prewarm to requestIdleCallback/setTimeout and cancels cleanly on AbortController', async () => {
  // 1. Uses requestIdleCallback when available: completes normally (true) and cancels handle on abort
  const idleDoneCtrl = new AbortController();
  let completedIdleCalled = false;
  const idleCompleteHost = {
    requestIdleCallback(cb, opts) {
      assert.equal(opts.timeout, 4000);
      queueMicrotask(() => {
        completedIdleCalled = true;
        cb({ didTimeout: false, timeRemaining: () => 50 });
      });
      return 7;
    },
    cancelIdleCallback() {},
    setTimeout: globalThis.setTimeout,
    clearTimeout: globalThis.clearTimeout,
  };
  assert.equal(await waitForIdleTurn(idleDoneCtrl.signal, idleCompleteHost, 4000, 100), true);
  assert.equal(completedIdleCalled, true);

  const cancelledIdleHandles = [];
  let pendingIdleCb = null;
  const idleHost = {
    requestIdleCallback(cb, opts) {
      assert.equal(opts.timeout, 4000);
      pendingIdleCb = cb;
      return 42;
    },
    cancelIdleCallback(handle) {
      cancelledIdleHandles.push(handle);
      pendingIdleCb = null;
    },
    setTimeout: globalThis.setTimeout,
    clearTimeout: globalThis.clearTimeout,
  };

  const abortCtrl = new AbortController();
  const waitPromise = waitForIdleTurn(abortCtrl.signal, idleHost, 4000, 100);
  assert.ok(pendingIdleCb !== null);
  abortCtrl.abort();
  assert.equal(await waitPromise, false);
  assert.deepEqual(cancelledIdleHandles, [42]);

  // 2. Falls back to setTimeout when requestIdleCallback is unavailable: completes normally and cancels cleanly
  const fallbackDoneCtrl = new AbortController();
  const fallbackCompleteHost = {
    setTimeout(cb, delay) {
      assert.equal(delay, 77);
      queueMicrotask(cb);
      return 101;
    },
    clearTimeout() {},
  };
  assert.equal(
    await waitForIdleTurn(fallbackDoneCtrl.signal, fallbackCompleteHost, 4000, 77),
    true,
  );

  const clearedTimers = [];
  const fallbackHost = {
    setTimeout(cb, delay) {
      assert.equal(delay, 123);
      return { id: 99, cb };
    },
    clearTimeout(handle) {
      clearedTimers.push(handle.id);
    },
  };
  const fallbackAbort = new AbortController();
  const fallbackWait = waitForIdleTurn(fallbackAbort.signal, fallbackHost, 4000, 123);
  fallbackAbort.abort();
  assert.equal(await fallbackWait, false);
  assert.deepEqual(clearedTimers, [99]);

  // 3. Staggered idle scheduling runs each target sequentially on idle turns and stops on abort
  const events = [];
  let nextHandle = 1;
  const scheduledHost = {
    requestIdleCallback(cb) {
      const id = nextHandle;
      nextHandle += 1;
      events.push(`idle_scheduled:${id}`);
      queueMicrotask(() => cb({ didTimeout: false, timeRemaining: () => 50 }));
      return id;
    },
    cancelIdleCallback() {},
    setTimeout(cb, delay) {
      events.push(`delay:${delay}`);
      queueMicrotask(cb);
      return nextHandle++;
    },
    clearTimeout() {},
  };

  const loadedSlugs = [];
  const seqAbort = new AbortController();
  await prewarmBenchmarkTargets({
    targets: [{ slug: 'wise_0855' }, { slug: 'barnards_star' }, { slug: 'comet_3i_atlas' }],
    signal: seqAbort.signal,
    host: scheduledHost,
    initialDelayMs: 50,
    staggerDelayMs: 25,
    loadPreset: async (slug) => {
      loadedSlugs.push(slug);
      if (slug === 'barnards_star') {
        seqAbort.abort();
      }
    },
  });

  assert.deepEqual(loadedSlugs, ['wise_0855', 'barnards_star']);
  assert.deepEqual(events, ['delay:50', 'idle_scheduled:2', 'delay:25', 'idle_scheduled:4']);

  // 3b. Fallback setTimeout-only host prewarms all targets and skips non-abort fetch errors
  const fallbackEvents = [];
  const fallbackLoaded = [];
  await prewarmBenchmarkTargets({
    targets: [{ slug: 'wise_0855' }, { slug: 'broken_slug' }, { slug: 'comet_3i_atlas' }],
    signal: new AbortController().signal,
    host: {
      setTimeout(cb, delay) {
        fallbackEvents.push(`timer:${delay}`);
        queueMicrotask(cb);
        return fallbackEvents.length;
      },
      clearTimeout() {},
    },
    initialDelayMs: 40,
    staggerDelayMs: 15,
    fallbackDelayMs: 10,
    loadPreset: async (slug) => {
      if (slug === 'broken_slug') {
        throw new Error('Transient 503');
      }
      fallbackLoaded.push(slug);
    },
  });
  assert.deepEqual(fallbackLoaded, ['wise_0855', 'comet_3i_atlas']);
  assert.deepEqual(fallbackEvents, [
    'timer:40',
    'timer:10',
    'timer:15',
    'timer:10',
    'timer:15',
    'timer:10',
  ]);

  // 3c. Aborting during initialDelayMs clears timer without scheduling idle or loading presets
  const initialCleared = [];
  let initialIdleCalled = false;
  let initialPresetCalled = false;
  const initialAbort = new AbortController();
  const initialWaitPromise = prewarmBenchmarkTargets({
    targets: [{ slug: 'wise_0855' }],
    signal: initialAbort.signal,
    host: {
      requestIdleCallback() {
        initialIdleCalled = true;
        return 1;
      },
      setTimeout(_cb, delay) {
        assert.equal(delay, 500);
        return 777;
      },
      clearTimeout(id) {
        initialCleared.push(id);
      },
    },
    initialDelayMs: 500,
    loadPreset: async () => {
      initialPresetCalled = true;
    },
  });
  initialAbort.abort();
  await initialWaitPromise;
  assert.deepEqual(initialCleared, [777]);
  assert.equal(initialIdleCalled, false);
  assert.equal(initialPresetCalled, false);

  // 3d. Default host resolution tolerates a partial globalThis.window without setTimeout
  const origWindowDesc = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {},
  });
  try {
    const partialWinAbort = new AbortController();
    const p = waitForIdleTurn(partialWinAbort.signal, undefined, 4000, 50);
    partialWinAbort.abort();
    assert.equal(await p, false);
  } finally {
    if (origWindowDesc) {
      Object.defineProperty(globalThis, 'window', origWindowDesc);
    } else {
      delete globalThis.window;
    }
  }

  // 4. Empty target list or pre-aborted signal skips all scheduling
  const preAborted = new AbortController();
  preAborted.abort();
  let calledWhenAborted = false;
  await prewarmBenchmarkTargets({
    targets: [{ slug: 'wise_0855' }],
    signal: preAborted.signal,
    host: scheduledHost,
    loadPreset: async () => {
      calledWhenAborted = true;
    },
  });
  await prewarmBenchmarkTargets({
    targets: [],
    signal: new AbortController().signal,
    host: scheduledHost,
    loadPreset: async () => {
      calledWhenAborted = true;
    },
  });
  assert.equal(calledWhenAborted, false);
});

test('Live coordinate loading skips /api/v1/spectrum prefetch, surfaces FastAPI error details, and times out cleanly', async () => {
  const manifest = readPresetManifest('wise_0855');
  const liveManifest = {
    ...manifest,
    epochs: manifest.epochs.map((ep, idx) => ({
      ...ep,
      binary_dtype: 'float16',
      detectors_data_url: `/api/v1/frames/live_grid/ep_${idx}_detectors.bin`,
      flags_mask_url: `/api/v1/frames/live_grid/ep_${idx}_flags.bin`,
    })),
    default_spectrum_url: '/api/v1/spectrum?ra=131.4500&dec=-11.1000',
  };

  const float16Frame = new Uint16Array(6 * 256 * 256);
  const flagsFrame = new Uint8Array(256 * 256);
  const requestedUrls = [];

  const mockLiveFetch = async (url) => {
    requestedUrls.push(url);
    if (url.startsWith('/api/v1/catalogs')) {
      return new Response(
        JSON.stringify({
          grid_key: 'live_grid',
          catalog_overlays: manifest.catalog_overlays,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
    if (url.startsWith('/api/v1/sky-patch')) {
      return new Response(JSON.stringify(liveManifest), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.includes('detectors.bin')) {
      return new Response(float16Frame.buffer, {
        status: 200,
        headers: { 'X-Binary-Dtype': 'float16' },
      });
    }
    if (url.includes('flags.bin')) {
      return new Response(flagsFrame.buffer, { status: 200 });
    }
    throw new Error(`Unexpected URL fetched during live load: ${url}`);
  };

  let readyPayload = null;
  await loadLiveCoordinatesFlow(131.45, -11.1, 5.0, {
    cache: new ClientPatchCache(4),
    fetchImpl: mockLiveFetch,
    onReady: (res) => {
      readyPayload = res;
    },
  });
  assert.ok(readyPayload);
  assert.equal(readyPayload.spectrum, null);
  assert.equal(
    requestedUrls.some((u) => u.includes('/api/v1/spectrum')),
    false,
    'Live coordinate load must not prefetch /api/v1/spectrum',
  );

  // FastAPI JSON error detail is surfaced directly
  await assert.rejects(
    () =>
      loadLiveCoordinatesFlow(131.45, -11.1, 5.0, {
        cache: new ClientPatchCache(4),
        fetchImpl: async (url) => {
          if (url.startsWith('/api/v1/sky-patch')) {
            return new Response(JSON.stringify({ detail: 'IRSA archive unavailable' }), {
              status: 503,
              headers: { 'Content-Type': 'application/json' },
            });
          }
          return new Response(null, { status: 503 });
        },
        onReady: () => {},
      }),
    /Failed to load sky patch \(503\): IRSA archive unavailable/,
  );

  // Client timeout rejects cleanly without hanging
  await assert.rejects(
    () =>
      loadLiveCoordinatesFlow(131.45, -11.1, 5.0, {
        cache: new ClientPatchCache(4),
        timeouts: { catalogsMs: 25, patchMs: 25, frameMs: 25 },
        fetchImpl: (_url, init) =>
          new Promise((_, reject) => {
            init?.signal?.addEventListener(
              'abort',
              () => {
                const err = new Error('Aborted');
                err.name = 'AbortError';
                reject(err);
              },
              { once: true },
            );
          }),
        onReady: () => {},
      }),
    /timed out/i,
  );
});

test('ClientPatchCache piggybacks X-Cache-Delta once on subsequent /api/v1/sky-patch request', async () => {
  const cache = new ClientPatchCache(4);
  const presetManifest = readPresetManifest('wise_0855');

  // Seed one cached catalog so we can trigger a local cache hit
  await cache.putCatalogs('ra10.00_dec20.00_fov5.0', {
    query: { ra: 10, dec: 20, fov_arcmin: 5, grid_key: 'ra10.00_dec20.00_fov5.0', cached: false },
    wcs: presetManifest.wcs,
    catalog_overlays: presetManifest.catalog_overlays,
  });
  await cache.getCatalogs('ra10.00_dec20.00_fov5.0');

  const seenPatchHeaders = [];
  const mockFetch = async (url, init) => {
    if (url.startsWith('/api/v1/sky-patch')) {
      seenPatchHeaders.push(init?.headers ?? null);
      return new Response(JSON.stringify(presetManifest), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.startsWith('/api/v1/catalogs')) {
      return new Response(
        JSON.stringify({
          query: presetManifest.query,
          wcs: presetManifest.wcs,
          catalog_overlays: presetManifest.catalog_overlays,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
    if (url.includes('_flags.bin')) {
      return new Response(new Uint8Array(65536), { status: 200 });
    }
    return new Response(new Uint8Array(1572864), {
      status: 200,
      headers: { 'X-Binary-Dtype': 'float32' },
    });
  };

  await loadLiveCoordinatesFlow(50, 10, 5.0, {
    cache,
    fetchImpl: mockFetch,
    onReady: () => {},
  });
  await loadLiveCoordinatesFlow(60, 15, 5.0, {
    cache,
    fetchImpl: mockFetch,
    onReady: () => {},
  });

  assert.deepEqual(seenPatchHeaders[0], { 'X-Cache-Delta': 'p=0,c=1' });
  assert.equal(seenPatchHeaders[1], null);
});
