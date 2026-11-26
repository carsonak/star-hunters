import type React from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { NotificationContextValue } from '../context/NotificationContext';
import { defaultClientPatchCache, livePatchCacheKey } from '../services/clientPatchCache';
import {
  loadLiveCoordinatesFlow,
  loadPresetPatchFlow,
  type LoadedPatchResult,
} from '../services/progressivePatchLoader';
import type {
  BinaryDtype,
  CatalogsResponse,
  FetchQualityMode,
  KnownCelestialObject,
  SkyExplorerContextValue,
  SkyPatchManifest,
} from '../types/contracts';
import {
  assignDeterministicHighlighterColors,
  getDefaultCheckedObjectIds,
} from '../utils/knownObjectSelection';
import {
  defaultNetworkQualityTracker,
  FAST_NETWORK_RECOVERY_MBPS,
  readStoredQualityMode,
  saveQualityMode,
} from '../utils/networkQuality';
import { observatoryLogger } from '../utils/observatoryLogger';
import { computeGridKey } from '../utils/wcsViewport';

/** Callbacks supplied to {@link usePatchLoader} to coordinate dependent context slices. */
export interface UsePatchLoaderOptions {
  /** Ref to the active notification context for async task toasts and warnings. */
  notificationsRef: React.RefObject<NotificationContextValue | null>;
  /** Invoked when a live load resets the active patch before fetching a new coordinate grid. */
  onResetPatchDependents: () => void;
  /** Invoked when Stage A catalogs arrive so default-checked object IDs can be synced. */
  onStageACheckedIds: (defaultCheckedIds: string[]) => void;
  /** Invoked when Stage B buffers and manifest are decoded and ready to render. */
  onStageBReady: (result: LoadedPatchResult, defaultCheckedIds: string[]) => void;
}

/**
 * Manage progressive Stage A (catalog) and Stage B (cutout binary frame) loading,
 * abort cancellation of stale navigations, and deterministic highlighter colors.
 *
 * @param options - Dependent slice synchronization callbacks and notification ref.
 * @returns Active preset slug, sky coordinates, patch buffers, loading state, and loaders.
 */
export function usePatchLoader({
  notificationsRef,
  onResetPatchDependents,
  onStageACheckedIds,
  onStageBReady,
}: UsePatchLoaderOptions) {
  const [activePresetSlug, setActivePresetSlug] = useState<string | null>('wise_0855');
  const [coordinates, setCoordinates] = useState<{
    ra: number;
    dec: number;
    fovArcmin: number;
  }>({
    ra: 133.7951,
    dec: -7.2451,
    fovArcmin: 5.0,
  });
  const [manifest, setManifest] = useState<SkyPatchManifest | null>(null);
  const [epochBuffers, setEpochBuffers] = useState<Float32Array[][] | null>(null);
  const [flagsBuffers, setFlagsBuffers] = useState<Uint8Array[] | null>(null);
  const [loadingState, setLoadingState] = useState<SkyExplorerContextValue['loadingState']>({
    stage: 'idle',
  });
  const [stageAKnownObjects, setStageAKnownObjects] = useState<KnownCelestialObject[] | null>(null);
  const [objectHighlighterColors, setObjectHighlighterColors] = useState<Record<string, string>>(
    {},
  );
  const [fetchQualityMode, setFetchQualityModeState] = useState<FetchQualityMode>(() =>
    readStoredQualityMode(),
  );
  const [activePatchDtype, setActivePatchDtype] = useState<BinaryDtype>('float32');
  const [networkSnapshot, setNetworkSnapshot] = useState(() =>
    defaultNetworkQualityTracker.getSnapshot(),
  );

  const fetchQualityModeRef = useRef<FetchQualityMode>(fetchQualityMode);
  const sessionRecommendedRef = useRef<Set<'compressed-good' | 'full-bad'>>(new Set());
  const setFetchQualityMode = useCallback((nextMode: FetchQualityMode) => {
    fetchQualityModeRef.current = nextMode;
    setFetchQualityModeState(nextMode);
    saveQualityMode(nextMode);
  }, []);

  const abortControllerRef = useRef<AbortController | null>(null);
  const lastLoadRequestRef = useRef<
    { kind: 'preset'; slug: string } | { kind: 'live'; ra: number; dec: number; fovArcmin: number }
  >({ kind: 'preset', slug: 'wise_0855' });
  const callbacksRef = useRef({
    onResetPatchDependents,
    onStageACheckedIds,
    onStageBReady,
  });
  useEffect(() => {
    callbacksRef.current = {
      onResetPatchDependents,
      onStageACheckedIds,
      onStageBReady,
    };
  }, [onResetPatchDependents, onStageACheckedIds, onStageBReady]);

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const knownObjects: KnownCelestialObject[] = useMemo(
    () => stageAKnownObjects ?? manifest?.catalog_overlays?.known_objects ?? [],
    [stageAKnownObjects, manifest],
  );

  const applyLoadedPatch = useCallback(
    (
      result: LoadedPatchResult,
      syncCoordinatesFromManifest: boolean,
      preserveStageAColors: boolean,
    ) => {
      const nextKnownObjects = result.manifest.catalog_overlays.known_objects ?? [];
      setStageAKnownObjects(null);
      setObjectHighlighterColors((prev) =>
        assignDeterministicHighlighterColors(nextKnownObjects, preserveStageAColors ? prev : {}),
      );
      setManifest(result.manifest);
      setEpochBuffers(result.epochBuffers);
      setFlagsBuffers(result.flagsBuffers);
      if (syncCoordinatesFromManifest) {
        setCoordinates({
          ra: result.manifest.query.ra,
          dec: result.manifest.query.dec,
          fovArcmin: result.manifest.query.fov_arcmin,
        });
      }

      callbacksRef.current.onStageBReady(result, getDefaultCheckedObjectIds(nextKnownObjects));
      setLoadingState({ stage: 'ready', fromClientCache: result.fromClientCache });
      const actualDtype: BinaryDtype = result.loadedDtype ?? 'float32';
      setActivePatchDtype(actualDtype);
      const snap = defaultNetworkQualityTracker.getSnapshot();
      setNetworkSnapshot(snap);
      notificationsRef.current?.finishAsyncTask('sky-patch-load');

      const mode = fetchQualityModeRef.current;
      if (
        mode === 'auto' &&
        actualDtype === 'float16' &&
        (!result.fromClientCache || result.degradedOnAutoFallback)
      ) {
        notificationsRef.current?.pushWarning({
          id: 'quality-auto-degraded',
          title: 'Reduced image quality (Compressed) due to slow connection',
          detail: 'Switch to Full or click Refetch in Blinker once your connection improves.',
          level: 'warning',
        });
      } else if (
        mode === 'compressed' &&
        !snap.isSlow &&
        snap.measuredMbps !== null &&
        snap.measuredMbps >= FAST_NETWORK_RECOVERY_MBPS &&
        !sessionRecommendedRef.current.has('compressed-good')
      ) {
        sessionRecommendedRef.current.add('compressed-good');
        notificationsRef.current?.pushWarning({
          id: 'quality-recommend-upgrade',
          title: `Fast connection detected (${snap.measuredMbps.toFixed(1)} Mbps)`,
          detail: 'Switch from Compressed to Auto or Full for full scientific precision.',
          level: 'warning',
          actions: [
            { label: 'Switch to Auto', onClick: () => setFetchQualityMode('auto') },
            { label: 'Switch to Full', onClick: () => setFetchQualityMode('full') },
          ],
        });
      } else if (mode === 'full' && snap.isSlow && !sessionRecommendedRef.current.has('full-bad')) {
        sessionRecommendedRef.current.add('full-bad');
        notificationsRef.current?.pushWarning({
          id: 'quality-recommend-downgrade',
          title: `Slow connection detected${snap.measuredMbps ? ` (${snap.measuredMbps.toFixed(1)} Mbps)` : ''}`,
          detail: 'Switch from Full to Auto or Compressed to improve load times.',
          level: 'warning',
          actions: [
            { label: 'Switch to Auto', onClick: () => setFetchQualityMode('auto') },
            { label: 'Switch to Compressed', onClick: () => setFetchQualityMode('compressed') },
          ],
        });
      }

      observatoryLogger.info('patch', 'Sky patch ready for blinking', {
        gridKey: result.manifest.query.grid_key,
        mission: result.manifest.data_source_summary?.primary_mission,
        epochs: result.manifest.epochs.length,
        fromClientCache: result.fromClientCache,
        dtype: actualDtype,
      });
    },
    [notificationsRef, setFetchQualityMode],
  );

  const runPatchLoad = useCallback(
    async (options: {
      taskMessage: string;
      errorTitle: string;
      fallbackErrorMessage: string;
      resetCurrentPatch?: boolean;
      logStart: () => void;
      logError: (errorMessage: string) => void;
      execute: (signal: AbortSignal) => Promise<void>;
    }) => {
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setStageAKnownObjects(null);
      if (options.resetCurrentPatch) {
        setObjectHighlighterColors({});
        setManifest(null);
        setEpochBuffers(null);
        setFlagsBuffers(null);
        callbacksRef.current.onResetPatchDependents();
      }
      options.logStart();
      notificationsRef.current?.startAsyncTask('sky-patch-load', options.taskMessage, 'progress');

      try {
        await options.execute(controller.signal);
      } catch (err) {
        if ((err as { name?: string } | null)?.name === 'AbortError' || controller.signal.aborted) {
          return;
        }
        notificationsRef.current?.finishAsyncTask('sky-patch-load');
        const msg = err instanceof Error ? err.message : options.fallbackErrorMessage;
        options.logError(msg);
        setLoadingState({ stage: 'error', message: msg });
        notificationsRef.current?.pushWarning({
          id: 'patch-load-error',
          title: options.errorTitle,
          detail: msg,
          level: 'error',
        });
      }
    },
    [notificationsRef],
  );

  const loadPresetBundle = useCallback(
    async (slug: string, bypassClientCache = false) => {
      lastLoadRequestRef.current = { kind: 'preset', slug };
      await runPatchLoad({
        taskMessage: 'Loading target frames...',
        errorTitle: 'Failed to load preset patch',
        fallbackErrorMessage: 'Unknown error loading preset',
        logStart: () =>
          observatoryLogger.info('patch', 'Loading benchmark preset bundle', { slug }),
        logError: (error) =>
          observatoryLogger.error('patch', 'Preset bundle load failed', { slug, error }),
        execute: async (signal) => {
          await loadPresetPatchFlow(slug, {
            cache: defaultClientPatchCache,
            signal,
            bypassClientCache,
            onStageChange: setLoadingState,
            onReady: (result) => applyLoadedPatch(result, true, false),
          });
        },
      });
    },
    [applyLoadedPatch, runPatchLoad],
  );

  const loadLiveCoordinates = useCallback(
    async (ra: number, dec: number, fovArcmin: number, bypassClientCache = false) => {
      lastLoadRequestRef.current = { kind: 'live', ra, dec, fovArcmin };
      await runPatchLoad({
        taskMessage: 'Streaming epoch frames...',
        errorTitle: 'Sky patch stream interrupted',
        fallbackErrorMessage: 'Unknown error loading sky patch',
        resetCurrentPatch: true,
        logStart: () =>
          observatoryLogger.info('patch', 'Loading live sky patch', { ra, dec, fovArcmin }),
        logError: (error) =>
          observatoryLogger.error('patch', 'Live sky patch stream interrupted', {
            ra,
            dec,
            error,
          }),
        execute: async (signal) => {
          await loadLiveCoordinatesFlow(ra, dec, fovArcmin, {
            cache: defaultClientPatchCache,
            signal,
            qualityMode: fetchQualityModeRef.current,
            networkTracker: defaultNetworkQualityTracker,
            bypassClientCache,
            onStageChange: setLoadingState,
            onCatalogsReady: (catData: CatalogsResponse) => {
              const stageAObjects = catData.catalog_overlays.known_objects ?? [];
              observatoryLogger.info('catalogs', 'Known objects loaded for patch', {
                count: stageAObjects.length,
                ra,
                dec,
              });
              setStageAKnownObjects(stageAObjects);
              setObjectHighlighterColors(assignDeterministicHighlighterColors(stageAObjects, {}));
              callbacksRef.current.onStageACheckedIds(getDefaultCheckedObjectIds(stageAObjects));
              notificationsRef.current?.startAsyncTask(
                'sky-patch-catalogs',
                'Known objects loaded',
                'success',
                2600,
              );
              notificationsRef.current?.startAsyncTask(
                'sky-patch-load',
                'Streaming epoch frames...',
                'progress',
              );
              setLoadingState({
                stage: 'loading_cutouts',
                message: 'Streaming epoch frames...',
              });
            },
            onReady: (result) => applyLoadedPatch(result, false, true),
          });
        },
      });
    },
    [applyLoadedPatch, notificationsRef, runPatchLoad],
  );

  const retryActivePatch = useCallback(() => {
    const req = lastLoadRequestRef.current;
    if (req.kind === 'preset') {
      void loadPresetBundle(req.slug);
    } else {
      void loadLiveCoordinates(req.ra, req.dec, req.fovArcmin);
    }
  }, [loadLiveCoordinates, loadPresetBundle]);

  const refetchActivePatch = useCallback(() => {
    const req = lastLoadRequestRef.current;
    if (req.kind === 'preset') {
      void defaultClientPatchCache.deletePatch(`preset:${req.slug}`).finally(() => {
        void loadPresetBundle(req.slug, true);
      });
    } else {
      const gk = computeGridKey(req.ra, req.dec, req.fovArcmin);
      void Promise.all([
        defaultClientPatchCache.deletePatch(livePatchCacheKey(gk, 'float32')),
        defaultClientPatchCache.deletePatch(livePatchCacheKey(gk, 'float16')),
        defaultClientPatchCache.deletePatch(gk),
      ]).finally(() => {
        void loadLiveCoordinates(req.ra, req.dec, req.fovArcmin, true);
      });
    }
  }, [loadLiveCoordinates, loadPresetBundle]);

  return {
    activePresetSlug,
    setActivePresetSlug,
    coordinates,
    setCoordinates,
    manifest,
    epochBuffers,
    flagsBuffers,
    loadingState,
    knownObjects,
    objectHighlighterColors,
    fetchQualityMode,
    setFetchQualityMode,
    activePatchDtype,
    isNetworkSlow: networkSnapshot.isSlow,
    measuredBandwidthMbps: networkSnapshot.measuredMbps,
    loadPresetBundle,
    loadLiveCoordinates,
    retryActivePatch,
    refetchActivePatch,
  };
}
