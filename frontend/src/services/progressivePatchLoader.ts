import type {
  BinaryDtype,
  CatalogsResponse,
  EpochManifestEntry,
  FetchQualityMode,
  SkyPatchManifest,
  SpectrumPayload,
} from '../types/contracts.ts';
import { decodeDetectorBinaryBuffer, decodeFlagsBinaryBuffer } from '../utils/binaryDecoder.ts';
import {
  AUTO_FLOAT32_STALL_MS,
  defaultNetworkQualityTracker,
  type NetworkQualityTracker,
} from '../utils/networkQuality.ts';
import { observatoryLogger } from '../utils/observatoryLogger.ts';
import { computeGridKey } from '../utils/wcsViewport.ts';
import { livePatchCacheKey, type ClientPatchCache } from './clientPatchCache.ts';

/** Progressive two-stage loading state emitted during preset and live sky patch loads. */
export type ProgressiveLoadingStage =
  | { stage: 'idle' }
  | { stage: 'loading_catalogs'; message: string }
  | { stage: 'loading_cutouts'; message: string }
  | { stage: 'ready'; fromClientCache: boolean }
  | { stage: 'error'; message: string };

/** Fully decoded sky patch payload delivered to {@link PresetFlowOptions.onReady} / {@link LiveFlowOptions.onReady}. */
export interface LoadedPatchResult {
  manifest: SkyPatchManifest;
  epochBuffers: Float32Array[][];
  flagsBuffers: Uint8Array[];
  spectrum: SpectrumPayload | null;
  fromClientCache: boolean;
  loadedDtype?: BinaryDtype;
  degradedOnAutoFallback?: boolean;
}

/** Terminal outcome of a progressive patch loading flow. */
export interface FlowOutcome {
  aborted: boolean;
  fromClientCache?: boolean;
  loadedDtype?: BinaryDtype;
}

/** Minimal `fetch` function signature accepted for dependency injection in tests. */
export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Options for {@link loadPresetPatchFlow}. */
export interface PresetFlowOptions {
  cache: ClientPatchCache;
  fetchImpl?: FetchLike;
  signal?: AbortSignal;
  bypassClientCache?: boolean;
  onStageChange?: (state: ProgressiveLoadingStage) => void;
  onReady: (result: LoadedPatchResult) => void;
}

/** Options for {@link loadLiveCoordinatesFlow}. */
export interface LiveFlowOptions {
  cache: ClientPatchCache;
  fetchImpl?: FetchLike;
  signal?: AbortSignal;
  qualityMode?: FetchQualityMode;
  networkTracker?: NetworkQualityTracker;
  bypassClientCache?: boolean;
  stallTimeoutMs?: number;
  onStageChange?: (state: ProgressiveLoadingStage) => void;
  onCatalogsReady?: (catalogs: CatalogsResponse) => void;
  onReady: (result: LoadedPatchResult) => void;
  timeouts?: {
    catalogsMs?: number;
    patchMs?: number;
    frameMs?: number;
  };
}

export const CATALOGS_FETCH_TIMEOUT_MS = 15_000;
export const SKY_PATCH_FETCH_TIMEOUT_MS = 28_000;
export const FRAME_FETCH_TIMEOUT_MS = 15_000;
export const PRESET_FETCH_TIMEOUT_MS = 15_000;

/** Options for {@link fetchAndCachePatchBundle}. */
interface FetchPatchBundleOptions {
  cache: ClientPatchCache;
  cacheKey: string;
  liveGridKey?: string;
  manifest: SkyPatchManifest;
  fetchImpl: FetchLike;
  signal?: AbortSignal;
  targetDtype: BinaryDtype;
  appendPrecisionQuery: boolean;
  autoFallbackToFloat16: boolean;
  stallTimeoutMs: number;
  networkTracker: NetworkQualityTracker;
  prefetchSpectrum: boolean;
  frameTimeoutMs?: number;
  onReady: (result: LoadedPatchResult) => void;
}

async function fetchWithTimeout(
  fetchImpl: FetchLike,
  input: string,
  signal: AbortSignal | undefined,
  timeoutMs: number,
  label: string,
): Promise<Response> {
  if (signal?.aborted) {
    const abortErr = new Error('Aborted');
    abortErr.name = 'AbortError';
    throw abortErr;
  }
  const controller = new AbortController();
  let timedOut = false;
  const onExternalAbort = () => controller.abort();
  signal?.addEventListener('abort', onExternalAbort, { once: true });
  const timerId = globalThis.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const res = await fetchImpl(input, { signal: controller.signal });
    if (signal?.aborted) {
      const abortErr = new Error('Aborted');
      abortErr.name = 'AbortError';
      throw abortErr;
    }
    return res;
  } catch (err) {
    if (timedOut && !signal?.aborted) {
      throw new Error(`${label} timed out after ${Math.round(timeoutMs / 1000)}s`, { cause: err });
    }
    throw err;
  } finally {
    globalThis.clearTimeout(timerId);
    signal?.removeEventListener('abort', onExternalAbort);
  }
}

async function readErrorDetail(res: Response, fallbackPrefix: string): Promise<string> {
  try {
    const contentType = res.headers?.get?.('content-type') || '';
    if (contentType.includes('application/json')) {
      const body = (await res.json()) as { detail?: unknown; message?: unknown };
      const msg =
        typeof body?.detail === 'string'
          ? body.detail
          : typeof body?.message === 'string'
            ? body.message
            : '';
      if (msg.trim()) {
        return `${fallbackPrefix} (${res.status}): ${msg.trim()}`;
      }
    }
  } catch {
    // Fall back to HTTP status string
  }
  return `${fallbackPrefix} (${res.status})`;
}

/**
 * Attempt to satisfy a patch load directly from the two-tier client cache.
 *
 * @param cache - Client patch cache instance.
 * @param cacheKey - Canonical cache key (`preset:{slug}` or quantized `grid_key`).
 * @param signal - Optional abort signal.
 * @param onReady - Callback invoked on cache hit.
 * @returns A resolved {@link FlowOutcome} on hit or abort, or `null` on cache miss.
 */
async function tryServeCachedPatch(
  cache: ClientPatchCache,
  cacheKeys: ReadonlyArray<string>,
  signal: AbortSignal | undefined,
  onReady: (result: LoadedPatchResult) => void,
): Promise<FlowOutcome | null> {
  for (const key of cacheKeys) {
    const cached = await cache.getPatch(key);
    if (signal?.aborted) return { aborted: true };
    if (cached) {
      const loadedDtype: BinaryDtype = cached.dtype ?? 'float32';
      onReady({
        manifest: cached.manifest,
        epochBuffers: cached.epochBuffers,
        flagsBuffers: cached.flagsBuffers,
        spectrum: cached.spectrum ?? null,
        fromClientCache: true,
        loadedDtype,
      });
      return { aborted: false, fromClientCache: true, loadedDtype };
    }
  }
  return null;
}

function buildDetectorFrameUrl(
  baseUrl: string,
  precision: BinaryDtype,
  appendQuery: boolean,
): string {
  if (!appendQuery || baseUrl.includes('?')) {
    return baseUrl;
  }
  return `${baseUrl}?precision=${precision}`;
}

async function fetchDetectorWithThroughput(
  fetchImpl: FetchLike,
  url: string,
  signal: AbortSignal | undefined,
  timeoutMs: number,
  networkTracker: NetworkQualityTracker,
): Promise<{ res: Response; buf: ArrayBuffer }> {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const res = await fetchWithTimeout(fetchImpl, url, signal, timeoutMs, 'Detector frame');
  if (!res.ok) {
    throw new Error(await readErrorDetail(res, 'Failed to load detector frame'));
  }
  const buf = await res.arrayBuffer();
  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const headerLen = Number(res.headers?.get?.('content-length'));
  const bytes = Number.isFinite(headerLen) && headerLen > 0 ? headerLen : buf.byteLength;
  networkTracker.recordTransfer(bytes, Math.max(1, endedAt - startedAt));
  return { res, buf };
}

/**
 * Fetch and decode the 6-detector binary cube and 8-bit hygiene flags mask for one epoch.
 */
async function fetchDecodedEpoch(
  epoch: EpochManifestEntry,
  options: {
    fetchImpl: FetchLike;
    signal?: AbortSignal;
    targetDtype: BinaryDtype;
    appendPrecisionQuery: boolean;
    autoFallbackToFloat16: boolean;
    stallTimeoutMs: number;
    networkTracker: NetworkQualityTracker;
    frameTimeoutMs?: number;
  },
): Promise<{
  detectors: Float32Array[];
  flags: Uint8Array;
  dtype: BinaryDtype;
  degraded: boolean;
}> {
  const {
    fetchImpl,
    signal,
    targetDtype,
    appendPrecisionQuery,
    autoFallbackToFloat16,
    stallTimeoutMs,
    networkTracker,
    frameTimeoutMs = FRAME_FETCH_TIMEOUT_MS,
  } = options;

  const flagPromise = fetchWithTimeout(
    fetchImpl,
    epoch.flags_mask_url,
    signal,
    frameTimeoutMs,
    'Flags frame',
  ).then(async (flagRes) => {
    if (!flagRes.ok) {
      throw new Error(await readErrorDetail(flagRes, 'Failed to load flags frame'));
    }
    return flagRes.arrayBuffer();
  });

  let detRes: Response;
  let detBuf: ArrayBuffer;
  let flagBuf: ArrayBuffer;
  let degraded = false;

  try {
    const primaryUrl = buildDetectorFrameUrl(
      epoch.detectors_data_url,
      targetDtype,
      appendPrecisionQuery,
    );
    if (autoFallbackToFloat16 && targetDtype === 'float32' && appendPrecisionQuery) {
      try {
        const result = await fetchDetectorWithThroughput(
          fetchImpl,
          primaryUrl,
          signal,
          Math.min(frameTimeoutMs, stallTimeoutMs),
          networkTracker,
        );
        detRes = result.res;
        detBuf = result.buf;
      } catch (err) {
        if (signal?.aborted) throw err;
        networkTracker.recordStall();
        degraded = true;
        observatoryLogger.warn(
          'patch_loader',
          'Auto-downgrading stalled float32 frame to float16',
          {
            url: primaryUrl,
            reason: err instanceof Error ? err.message : 'stall',
          },
        );
        const fallbackUrl = buildDetectorFrameUrl(epoch.detectors_data_url, 'float16', true);
        const fallback = await fetchDetectorWithThroughput(
          fetchImpl,
          fallbackUrl,
          signal,
          frameTimeoutMs,
          networkTracker,
        );
        detRes = fallback.res;
        detBuf = fallback.buf;
      }
    } else {
      const result = await fetchDetectorWithThroughput(
        fetchImpl,
        primaryUrl,
        signal,
        frameTimeoutMs,
        networkTracker,
      );
      detRes = result.res;
      detBuf = result.buf;
    }

    flagBuf = await flagPromise;
  } catch (err) {
    await flagPromise.catch(() => undefined);
    throw err;
  }
  if (signal?.aborted) {
    const abortErr = new Error('Aborted');
    abortErr.name = 'AbortError';
    throw abortErr;
  }
  const headerDtype = (detRes.headers?.get?.('X-Binary-Dtype') ||
    (degraded ? 'float16' : targetDtype) ||
    epoch.binary_dtype ||
    'float32') as BinaryDtype;

  return {
    detectors: decodeDetectorBinaryBuffer(detBuf, headerDtype),
    flags: decodeFlagsBinaryBuffer(flagBuf),
    dtype: headerDtype,
    degraded,
  };
}

/**
 * Concurrently fetch and decode all epoch binary buffers and optional default
 * spectrum for a manifest, persist the bundle in {@link ClientPatchCache}, and
 * invoke `onReady`.
 */
async function fetchAndCachePatchBundle(options: FetchPatchBundleOptions): Promise<FlowOutcome> {
  const {
    cache,
    cacheKey,
    liveGridKey,
    manifest,
    fetchImpl,
    signal,
    targetDtype,
    appendPrecisionQuery,
    autoFallbackToFloat16,
    stallTimeoutMs,
    networkTracker,
    prefetchSpectrum,
    frameTimeoutMs,
    onReady,
  } = options;

  const epochPromises = manifest.epochs.map((epoch) =>
    fetchDecodedEpoch(epoch, {
      fetchImpl,
      signal,
      targetDtype,
      appendPrecisionQuery,
      autoFallbackToFloat16,
      stallTimeoutMs,
      networkTracker,
      frameTimeoutMs,
    }),
  );
  const spectrumPromise =
    prefetchSpectrum && manifest.default_spectrum_url
      ? fetchWithTimeout(
          fetchImpl,
          manifest.default_spectrum_url,
          signal,
          PRESET_FETCH_TIMEOUT_MS,
          'Preset spectrum',
        )
          .then(async (res) => (res.ok ? ((await res.json()) as SpectrumPayload) : null))
          .catch(() => null)
      : Promise.resolve(null);

  const [decodedEpochs, loadedSpectrum] = await Promise.all([
    Promise.all(epochPromises),
    spectrumPromise,
  ]);
  if (signal?.aborted) return { aborted: true };

  const epochBuffers = decodedEpochs.map((entry) => entry.detectors);
  const flagsBuffers = decodedEpochs.map((entry) => entry.flags);
  const actualDtype: BinaryDtype = decodedEpochs.some((e) => e.dtype === 'float16')
    ? 'float16'
    : 'float32';
  const degradedOnAutoFallback = decodedEpochs.some((e) => e.degraded);
  const resolvedCacheKey = liveGridKey ? livePatchCacheKey(liveGridKey, actualDtype) : cacheKey;

  const now = Date.now();
  await cache.putPatch({
    cacheKey: resolvedCacheKey,
    dtype: actualDtype,
    manifest,
    epochBuffers,
    flagsBuffers,
    spectrum: loadedSpectrum,
    storedAt: now,
    lastAccessed: now,
  });
  if (signal?.aborted) return { aborted: true };

  onReady({
    manifest,
    epochBuffers,
    flagsBuffers,
    spectrum: loadedSpectrum,
    fromClientCache: false,
    loadedDtype: actualDtype,
    degradedOnAutoFallback,
  });
  return { aborted: false, fromClientCache: false, loadedDtype: actualDtype };
}

/**
 * Load a static benchmark preset bundle (`/presets/{slug}/manifest.json` plus
 * epoch binary frames and default spectrum), serving from {@link ClientPatchCache}
 * when already cached.
 *
 * @param slug - Benchmark preset identifier (e.g., `'wise_0855'`).
 * @param options - Cache instance, abort signal, stage callback, and completion callback.
 * @returns Outcome indicating whether the flow was aborted or served from cache.
 * @throws {Error} When the manifest or any epoch binary frame fails to load.
 */
export async function loadPresetPatchFlow(
  slug: string,
  options: PresetFlowOptions,
): Promise<FlowOutcome> {
  const {
    cache,
    fetchImpl = fetch,
    signal,
    bypassClientCache = false,
    onStageChange,
    onReady,
  } = options;
  const cacheKey = `preset:${slug}`;

  if (!bypassClientCache) {
    const cacheOutcome = await tryServeCachedPatch(cache, [cacheKey], signal, onReady);
    if (cacheOutcome) return cacheOutcome;
  }

  onStageChange?.({ stage: 'loading_cutouts', message: `Loading preset ${slug}...` });
  const manifestRes = await fetchWithTimeout(
    fetchImpl,
    `/presets/${slug}/manifest.json`,
    signal,
    PRESET_FETCH_TIMEOUT_MS,
    'Preset manifest',
  );
  if (signal?.aborted) return { aborted: true };
  if (!manifestRes.ok) {
    throw new Error(await readErrorDetail(manifestRes, 'Failed to load preset manifest'));
  }
  const loadedManifest = (await manifestRes.json()) as SkyPatchManifest;
  if (signal?.aborted) return { aborted: true };

  return fetchAndCachePatchBundle({
    cache,
    cacheKey,
    manifest: loadedManifest,
    fetchImpl,
    signal,
    targetDtype: 'float32',
    appendPrecisionQuery: false,
    autoFallbackToFloat16: false,
    stallTimeoutMs: AUTO_FLOAT32_STALL_MS,
    networkTracker: defaultNetworkQualityTracker,
    prefetchSpectrum: true,
    onReady,
  });
}

/**
 * Load a live sky patch at `(ra, dec, fovArcmin)` using a two-stage progressive
 * pipeline and adaptive `float32`/`float16` precision selection.
 */
export async function loadLiveCoordinatesFlow(
  ra: number,
  dec: number,
  fovArcmin: number,
  options: LiveFlowOptions,
): Promise<FlowOutcome> {
  const {
    cache,
    fetchImpl = fetch,
    signal,
    qualityMode = 'auto',
    networkTracker = defaultNetworkQualityTracker,
    bypassClientCache = false,
    stallTimeoutMs = AUTO_FLOAT32_STALL_MS,
    onStageChange,
    onCatalogsReady,
    onReady,
    timeouts,
  } = options;
  const catalogsTimeoutMs = timeouts?.catalogsMs ?? CATALOGS_FETCH_TIMEOUT_MS;
  const patchTimeoutMs = timeouts?.patchMs ?? SKY_PATCH_FETCH_TIMEOUT_MS;
  const frameTimeoutMs = timeouts?.frameMs ?? FRAME_FETCH_TIMEOUT_MS;
  const gridKey = computeGridKey(ra, dec, fovArcmin);
  const targetDtype = networkTracker.resolveTargetDtype(qualityMode);

  if (!bypassClientCache) {
    const candidateKeys =
      targetDtype === 'float16'
        ? [livePatchCacheKey(gridKey, 'float32'), livePatchCacheKey(gridKey, 'float16'), gridKey]
        : [livePatchCacheKey(gridKey, 'float32'), gridKey];
    const cacheOutcome = await tryServeCachedPatch(cache, candidateKeys, signal, onReady);
    if (cacheOutcome) {
      if (!cacheOutcome.aborted) {
        observatoryLogger.info('patch_loader', 'Live patch cache hit', {
          gridKey,
          dtype: cacheOutcome.loadedDtype,
        });
      }
      return cacheOutcome;
    }
  }

  observatoryLogger.info('patch_loader', 'Fetching live sky patch & catalogs', {
    gridKey,
    ra,
    dec,
    fovArcmin,
  });
  onStageChange?.({
    stage: 'loading_catalogs',
    message: 'Querying sky catalogs...',
  });
  const qs = `ra=${encodeURIComponent(ra)}&dec=${encodeURIComponent(dec)}&fov=${encodeURIComponent(fovArcmin)}`;

  // Stage A (fast async catalogs) and Stage B (cutout manifest) run concurrently
  const catalogsPromise = fetchWithTimeout(
    fetchImpl,
    `/api/v1/catalogs?${qs}`,
    signal,
    catalogsTimeoutMs,
    'Catalogs lookup',
  )
    .then(async (res) => {
      if (signal?.aborted || !res.ok) return null;
      const catData = (await res.json()) as CatalogsResponse;
      if (signal?.aborted) return null;
      await cache.putCatalogs(gridKey, catData);
      if (signal?.aborted) return null;
      onCatalogsReady?.(catData);
      return catData;
    })
    .catch(() => null);

  const patchRes = await fetchWithTimeout(
    fetchImpl,
    `/api/v1/sky-patch?${qs}`,
    signal,
    patchTimeoutMs,
    'Sky patch request',
  );
  if (signal?.aborted) return { aborted: true };
  if (!patchRes.ok) {
    throw new Error(await readErrorDetail(patchRes, 'Failed to load sky patch'));
  }
  const loadedManifest = (await patchRes.json()) as SkyPatchManifest;
  if (signal?.aborted) return { aborted: true };

  await catalogsPromise;
  if (signal?.aborted) return { aborted: true };

  return fetchAndCachePatchBundle({
    cache,
    cacheKey: livePatchCacheKey(gridKey, targetDtype),
    liveGridKey: gridKey,
    manifest: loadedManifest,
    fetchImpl,
    signal,
    targetDtype,
    appendPrecisionQuery: true,
    autoFallbackToFloat16: qualityMode === 'auto',
    stallTimeoutMs,
    networkTracker,
    prefetchSpectrum: false,
    frameTimeoutMs,
    onReady,
  });
}
