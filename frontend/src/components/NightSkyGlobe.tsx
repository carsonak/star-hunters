import React, { useCallback, useEffect, useRef, useState } from 'react';
import { IAU_CONSTELLATIONS } from '../data/constellations';
import { useSkyExplorer, useSkyFrustum } from '../context/SkyExplorerContext';
import {
  clampGlobeFov,
  IRSA_2MASS_HIPS_URL,
  IRSA_WISE_HIPS_URL,
  PANEL_GRID_FOV_DEG,
  PANEL_GRID_STEP_DEG,
  resolveBestDeepSkyHiPS,
  resolveGlobeClickAction,
  resolveGlobeKeyboardAction,
  resolveGlobeLodTier,
  stepPanCoordinates,
  targetFovForClickAction,
  type DeepHiPSResolution,
  type GlobeProjection,
} from '../utils/globeNavigation';
import { observatoryLogger } from '../utils/observatoryLogger';
import {
  angularSeparationDeg,
  BENCHMARK_TARGET_OPTIONS,
  computeSpherExReticleCorners,
  formatRaDecCompact,
  getMarkerShapeKind,
  wrapRaDeg,
} from '../utils/observatoryMath';
import { quantizeGridCenter } from '../utils/wcsViewport';
import {
  attachAladinIdleThrottler,
  loadAladinLite,
  projectSkySin,
  unprojectSkySin,
  type AladinIdleThrottler,
  type AladinInstance,
} from '../utils/skyProjection';
import { FallbackSkyCanvas } from './globe/FallbackSkyCanvas';
import { GlobeControlsDock } from './globe/GlobeControlsDock';
import {
  GlobeOverlaySvg,
  type GlobeKnownMarker,
  type GlobeOverlayGeometry,
} from './globe/GlobeOverlaySvg';
import { GlobeTelemetryBar } from './globe/GlobeTelemetryBar';

const SPHEREX_PATCH_ARCMIN = 5.0;
const CURSOR_TOOLTIP_DELAY_MS = 450;

function isInteractiveOverlayTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return Boolean(
    el.closest(
      '.aladin-logo, .aladin-attribution, .aladin-box, .hud-panel, .globe-controls-dock, .globe-telemetry-chip, button, input, select, a',
    ),
  );
}

/**
 * Full-bleed interactive 3D Night-Sky Globe powered by Aladin Lite v3 with a
 * deterministic pure-black 2D canvas star-sphere fallback, SVG constellation &
 * catalog overlays, progressive click-to-zoom, and accessible pan/zoom dock.
 *
 * @returns Rendered sky globe layer, navigation dock, and telemetry bar.
 */
export const NightSkyGlobe: React.FC = () => {
  const {
    coordinates,
    knownObjects = [],
    globalKnownObjects = [],
    renderSettings,
    objectHighlighterColors = {},
    isBlinkerModalOpen = false,
    openBlinkerModal,
    fetchQualityMode = 'auto',
    setFetchQualityMode,
  } = useSkyExplorer();
  const { skyFrustum, setSkyFrustum, globeFocusRequest } = useSkyFrustum();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const aladinHostRef = useRef<HTMLDivElement | null>(null);
  const aladinInstanceRef = useRef<AladinInstance | null>(null);
  const aladinThrottlerRef = useRef<AladinIdleThrottler | null>(null);
  const isBlinkerModalOpenRef = useRef<boolean>(isBlinkerModalOpen);

  const [aladinReady, setAladinReady] = useState(false);
  const [aladinFailed, setAladinFailed] = useState(false);
  const [showConstellations, setShowConstellations] = useState<boolean>(true);
  const [projection, setProjection] = useState<GlobeProjection>('SIN');
  const [activeSurveyLabel, setActiveSurveyLabel] = useState<string>('AllWISE Wide (IRSA)');
  const [viewportSize, setViewportSize] = useState<{ width: number; height: number }>({
    width: 1280,
    height: 800,
  });
  const [cursorTooltip, setCursorTooltip] = useState<{
    x: number;
    y: number;
    ra: number;
    dec: number;
  } | null>(null);
  const [hoverPatchPoints, setHoverPatchPoints] = useState<string | null>(null);
  const [overlayGeometry, setOverlayGeometry] = useState<GlobeOverlayGeometry>({
    constellationLines: [],
    constellationLabels: [],
    knownMarkers: [],
  });

  const viewStateRef = useRef<{ ra: number; dec: number; fovDeg: number }>({
    ra: skyFrustum?.ra ?? coordinates.ra,
    dec: skyFrustum?.dec ?? coordinates.dec,
    fovDeg: skyFrustum?.fovDeg ?? 70,
  });
  const currentSurveyUrlRef = useRef<string>(IRSA_WISE_HIPS_URL);
  const deepHiPSCacheRef = useRef<DeepHiPSResolution | null>(null);
  const deepHiPSPromiseRef = useRef<Promise<DeepHiPSResolution> | null>(null);
  const cursorHoverTimerRef = useRef<number | null>(null);
  const dragRef = useRef({
    active: false,
    moved: false,
    startX: 0,
    startY: 0,
    startRa: coordinates.ra,
    startDec: coordinates.dec,
  });

  useEffect(() => {
    const host = containerRef.current;
    if (!host) return undefined;
    const updateSize = () => {
      aladinThrottlerRef.current?.wake();
      const rect = host.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setViewportSize({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateSize);
      return () => window.removeEventListener('resize', updateSize);
    }
    const ro = new ResizeObserver(updateSize);
    ro.observe(host);
    return () => ro.disconnect();
  }, []);

  const switchSurveyLayer = useCallback((url: string, label: string) => {
    setActiveSurveyLabel(label);
    if (currentSurveyUrlRef.current !== url) {
      currentSurveyUrlRef.current = url;
      aladinThrottlerRef.current?.wake();
      try {
        aladinInstanceRef.current?.setImageSurvey?.(url);
      } catch (err) {
        observatoryLogger.warn('globe', 'Aladin setImageSurvey failed', {
          survey: label,
          error: err instanceof Error ? err.message : 'unknown',
        });
      }
      observatoryLogger.info('globe', 'Globe background sky survey switched', {
        survey: label,
        fovDeg: Number(viewStateRef.current.fovDeg.toFixed(2)),
      });
    }
  }, []);

  const updateLodForFov = useCallback(
    (fovDeg: number) => {
      const tier = resolveGlobeLodTier(fovDeg);
      if (tier === 'allwise') {
        switchSurveyLayer(IRSA_WISE_HIPS_URL, 'AllWISE Wide (IRSA)');
        return;
      }
      if (tier === '2mass') {
        switchSurveyLayer(IRSA_2MASS_HIPS_URL, '2MASS High-Res (IRSA)');
        return;
      }
      if (deepHiPSCacheRef.current) {
        switchSurveyLayer(deepHiPSCacheRef.current.url, deepHiPSCacheRef.current.label);
        return;
      }
      if (!deepHiPSPromiseRef.current) {
        deepHiPSPromiseRef.current = resolveBestDeepSkyHiPS().then((resolved) => {
          deepHiPSCacheRef.current = resolved;
          return resolved;
        });
      }
      deepHiPSPromiseRef.current
        .then((resolved) => {
          if (resolveGlobeLodTier(viewStateRef.current.fovDeg) === 'spherex_deep') {
            switchSurveyLayer(resolved.url, resolved.label);
          }
        })
        .catch(() => {
          if (resolveGlobeLodTier(viewStateRef.current.fovDeg) === 'spherex_deep') {
            switchSurveyLayer(IRSA_2MASS_HIPS_URL, '2MASS High-Res (IRSA)');
          }
        });
    },
    [switchSurveyLayer],
  );

  const applyCameraState = useCallback(
    (nextRa: number, nextDec: number, nextFovDeg: number, driveAladin = true) => {
      const clampedFov = clampGlobeFov(nextFovDeg);
      const wrappedRa = wrapRaDeg(nextRa);
      const clampedDec = Math.max(-89.9, Math.min(89.9, nextDec));
      viewStateRef.current = { ra: wrappedRa, dec: clampedDec, fovDeg: clampedFov };
      updateLodForFov(clampedFov);
      const rect = containerRef.current?.getBoundingClientRect();
      setSkyFrustum?.({
        ra: wrappedRa,
        dec: clampedDec,
        fovDeg: clampedFov,
        aspect: rect && rect.height > 0 ? rect.width / rect.height : 1.4,
      });
      if (driveAladin && aladinInstanceRef.current) {
        aladinThrottlerRef.current?.wake();
        try {
          aladinInstanceRef.current.gotoRaDec(wrappedRa, clampedDec);
          aladinInstanceRef.current.setFoV?.(clampedFov);
        } catch (err) {
          observatoryLogger.warn('globe', 'Aladin camera update failed', {
            error: err instanceof Error ? err.message : 'unknown',
          });
        }
      }
    },
    [setSkyFrustum, updateLodForFov],
  );

  useEffect(() => {
    if (!globeFocusRequest) return undefined;
    const raf = window.requestAnimationFrame(() => {
      applyCameraState(globeFocusRequest.ra, globeFocusRequest.dec, globeFocusRequest.fovDeg, true);
      observatoryLogger.info('globe', 'Focused sky camera', {
        ra: Number(globeFocusRequest.ra.toFixed(4)),
        dec: Number(globeFocusRequest.dec.toFixed(4)),
        fovDeg: globeFocusRequest.fovDeg,
      });
    });
    return () => window.cancelAnimationFrame(raf);
  }, [applyCameraState, globeFocusRequest]);

  useEffect(() => {
    let cancelled = false;
    const hostEl = aladinHostRef.current;
    observatoryLogger.info('globe', 'Loading Aladin Lite v3 runtime from jsDelivr');
    loadAladinLite()
      .then((api) => {
        if (cancelled || !hostEl) return;
        hostEl.replaceChildren();
        const initState = viewStateRef.current;
        const instance = api.aladin(hostEl, {
          survey: IRSA_WISE_HIPS_URL,
          projection: 'SIN',
          cooFrame: 'ICRSd',
          target: `${initState.ra} ${initState.dec}`,
          fov: initState.fovDeg,
          showCooGrid: false,
          showCooGridControl: false,
          showFullscreenControl: false,
          showLayersControl: false,
          showGotoControl: false,
          showReticle: false,
          showFrame: false,
          showZoomControl: false,
          showFov: false,
          showProjectionControl: false,
          showCooLocation: false,
          showShareControl: false,
          showSettingsControl: false,
          showSimbadPointerControl: false,
          showStatusBar: false,
          showContextMenu: false,
          log: false,
          backgroundColor: '#000000',
        });
        aladinInstanceRef.current = instance;
        const throttler = attachAladinIdleThrottler(instance);
        aladinThrottlerRef.current = throttler;
        if (isBlinkerModalOpenRef.current || document.hidden) {
          throttler.setPaused(true);
        }
        window.dispatchEvent(new Event('resize'));
        window.setTimeout(() => {
          if (!cancelled) {
            throttler.wake();
            window.dispatchEvent(new Event('resize'));
          }
        }, 120);
        setAladinReady(true);
        observatoryLogger.info('globe', 'Aladin Lite v3 initialized cleanly', {
          projection: 'SIN',
          survey: 'allWISE',
        });

        const syncFromAladin = () => {
          throttler.wake();
          let pos: [number, number] | undefined;
          let rawFov: [number, number] | number | undefined;
          try {
            pos = instance.getRaDec?.();
            rawFov = instance.getFoV?.();
          } catch {
            return;
          }
          const parsedFov = Array.isArray(rawFov)
            ? Number(rawFov[0])
            : typeof rawFov === 'number'
              ? rawFov
              : NaN;
          if (!Number.isFinite(parsedFov) || parsedFov <= 0) return;
          const fovDeg = clampGlobeFov(parsedFov);
          const ra = pos ? wrapRaDeg(Number(pos[0])) : viewStateRef.current.ra;
          const dec = pos ? Number(pos[1]) : viewStateRef.current.dec;
          if (!Number.isFinite(ra) || !Number.isFinite(dec)) return;
          const prev = viewStateRef.current;
          if (
            Math.abs(prev.ra - ra) < 1e-4 &&
            Math.abs(prev.dec - dec) < 1e-4 &&
            Math.abs(prev.fovDeg - fovDeg) < 1e-4
          ) {
            return;
          }
          viewStateRef.current = { ra, dec, fovDeg };
          updateLodForFov(fovDeg);

          const rect = containerRef.current?.getBoundingClientRect();
          setSkyFrustum?.({
            ra,
            dec,
            fovDeg,
            aspect: rect && rect.height > 0 ? rect.width / rect.height : 1.4,
          });
        };

        instance.on?.('positionChanged', syncFromAladin);
        instance.on?.('zoomChanged', syncFromAladin);
      })
      .catch((err) => {
        if (!cancelled) {
          setAladinFailed(true);
          observatoryLogger.warn(
            'globe',
            'Aladin Lite unavailable; keeping pure-black 3D star sphere active',
            { error: err instanceof Error ? err.message : 'unavailable' },
          );
        }
      });

    return () => {
      cancelled = true;
      aladinThrottlerRef.current?.destroy();
      aladinThrottlerRef.current = null;
      aladinInstanceRef.current = null;
      hostEl?.replaceChildren();
    };
    // Intentionally initialize Aladin host once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const projectWorldPoint = useCallback(
    (ra: number, dec: number, width: number, height: number): { px: number; py: number } | null => {
      if (aladinReady && !aladinFailed && aladinInstanceRef.current?.world2pix) {
        try {
          const px = aladinInstanceRef.current.world2pix(ra, dec);
          if (px && Number.isFinite(px[0]) && Number.isFinite(px[1])) {
            return { px: px[0], py: px[1] };
          }
          return null;
        } catch {
          // Fall back to analytic SIN projection if Aladin WASM rejects transient coordinates
        }
      }
      const view = viewStateRef.current;
      const p = projectSkySin(ra, dec, view.ra, view.dec, view.fovDeg, width, height);
      return p.visible ? { px: p.px, py: p.py } : null;
    },
    [aladinFailed, aladinReady],
  );

  const screenToWorldCoords = (clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (aladinReady && !aladinFailed && aladinInstanceRef.current?.pix2world) {
      try {
        const res = aladinInstanceRef.current.pix2world(x, y);
        if (res && Number.isFinite(res[0]) && Number.isFinite(res[1])) {
          return { world: { ra: wrapRaDeg(res[0]), dec: res[1] }, rect };
        }
        return null;
      } catch {
        // Fall back to analytic SIN inverse projection if Aladin pix2world throws transiently
      }
    }
    const world = unprojectSkySin(
      x,
      y,
      viewStateRef.current.ra,
      viewStateRef.current.dec,
      viewStateRef.current.fovDeg,
      rect.width,
      rect.height,
    );
    return world ? { world, rect } : null;
  };

  const clearCursorTooltipTimer = useCallback(() => {
    if (cursorHoverTimerRef.current !== null) {
      window.clearTimeout(cursorHoverTimerRef.current);
      cursorHoverTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => clearCursorTooltipTimer(), [clearCursorTooltipTimer]);

  const liveFrustum = skyFrustum ?? {
    ra: coordinates.ra,
    dec: coordinates.dec,
    fovDeg: 70,
  };

  useEffect(() => {
    const raf = window.requestAnimationFrame(() => {
      if (isBlinkerModalOpen) {
        setOverlayGeometry({ constellationLines: [], constellationLabels: [], knownMarkers: [] });
        return;
      }

      const w = viewportSize.width;
      const h = viewportSize.height;
      const nextLines: GlobeOverlayGeometry['constellationLines'] = [];
      const nextLabels: GlobeOverlayGeometry['constellationLabels'] = [];
      if (showConstellations) {
        for (const c of IAU_CONSTELLATIONS) {
          c.segments.forEach((seg, idx) => {
            const p1 = projectWorldPoint(seg.ra1, seg.dec1, w, h);
            const p2 = projectWorldPoint(seg.ra2, seg.dec2, w, h);
            if (!p1 || !p2 || Math.hypot(p1.px - p2.px, p1.py - p2.py) > w * 0.85) return;
            if (
              Math.max(p1.px, p2.px) < -40 ||
              Math.min(p1.px, p2.px) > w + 40 ||
              Math.max(p1.py, p2.py) < -40 ||
              Math.min(p1.py, p2.py) > h + 40
            ) {
              return;
            }
            nextLines.push({
              key: `${c.abbr}-${idx}`,
              x1: p1.px,
              y1: p1.py,
              x2: p2.px,
              y2: p2.py,
            });
          });
          if (liveFrustum.fovDeg >= 10) {
            const centerPt = projectWorldPoint(c.centerRa, c.centerDec, w, h);
            if (
              centerPt &&
              centerPt.px >= 24 &&
              centerPt.px <= w - 24 &&
              centerPt.py >= 24 &&
              centerPt.py <= h - 24
            ) {
              nextLabels.push({ abbr: c.abbr, name: c.name, px: centerPt.px, py: centerPt.py });
            }
          }
        }
      }

      const checkedSet = new Set(renderSettings.checkedObjectIds);
      const nextKnownMarkers: GlobeKnownMarker[] = [];
      if (checkedSet.size > 0) {
        const seenIds = new Set<string>();
        for (const obj of [...knownObjects, ...globalKnownObjects]) {
          if (!checkedSet.has(obj.object_id) || seenIds.has(obj.object_id)) continue;
          seenIds.add(obj.object_id);
          const pt = projectWorldPoint(obj.ra, obj.dec, w, h);
          if (!pt || pt.px < 8 || pt.px > w - 8 || pt.py < 8 || pt.py > h - 8) continue;
          nextKnownMarkers.push({
            id: obj.object_id,
            label: obj.label,
            px: pt.px,
            py: pt.py,
            shape: getMarkerShapeKind(obj, false) === 'diamond' ? 'diamond' : 'triangle',
            color:
              objectHighlighterColors[obj.object_id] ??
              (obj.object_type === 'solar_system' ? '#fbbf24' : '#38bdf8'),
          });
          if (nextKnownMarkers.length >= 120) break;
        }
      }

      setOverlayGeometry({
        constellationLines: nextLines,
        constellationLabels: nextLabels,
        knownMarkers: nextKnownMarkers,
      });
    });

    return () => window.cancelAnimationFrame(raf);
  }, [
    aladinFailed,
    aladinReady,
    globalKnownObjects,
    isBlinkerModalOpen,
    knownObjects,
    liveFrustum.dec,
    liveFrustum.fovDeg,
    liveFrustum.ra,
    objectHighlighterColors,
    projectWorldPoint,
    projection,
    renderSettings.checkedObjectIds,
    showConstellations,
    viewportSize.height,
    viewportSize.width,
  ]);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || isBlinkerModalOpen || isInteractiveOverlayTarget(e.target)) return;
    aladinThrottlerRef.current?.wake();
    containerRef.current?.focus({ preventScroll: true });
    clearCursorTooltipTimer();
    setCursorTooltip(null);
    dragRef.current = {
      active: true,
      moved: false,
      startX: e.clientX,
      startY: e.clientY,
      startRa: viewStateRef.current.ra,
      startDec: viewStateRef.current.dec,
    };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isBlinkerModalOpen) return;
    if (dragRef.current.active) {
      aladinThrottlerRef.current?.wake();
    }
    clearCursorTooltipTimer();
    setCursorTooltip(null);

    if (isInteractiveOverlayTarget(e.target)) {
      setHoverPatchPoints(null);
      return;
    }

    const drag = dragRef.current;
    if (drag.active && (!aladinReady || aladinFailed)) {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      const w = containerRef.current?.getBoundingClientRect()?.width || window.innerWidth;
      const fov = viewStateRef.current.fovDeg;
      const degPerPx = fov / Math.max(1, w);
      const cosDec = Math.max(0.2, Math.cos((drag.startDec * Math.PI) / 180));
      applyCameraState(
        wrapRaDeg(drag.startRa + (dx * degPerPx) / cosDec),
        Math.max(-89.9, Math.min(89.9, drag.startDec + dy * degPerPx)),
        fov,
        false,
      );
    } else if (drag.active && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 5) {
      drag.moved = true;
    }

    const hit = screenToWorldCoords(e.clientX, e.clientY);
    if (!hit) {
      setHoverPatchPoints(null);
      return;
    }

    const { raQ, decQ } = quantizeGridCenter(hit.world.ra, hit.world.dec, PANEL_GRID_STEP_DEG);
    const corners = computeSpherExReticleCorners(raQ, decQ, SPHEREX_PATCH_ARCMIN);
    const pts: string[] = [];
    for (const [cRa, cDec] of corners) {
      const proj = projectWorldPoint(cRa, cDec, hit.rect.width, hit.rect.height);
      if (!proj) break;
      pts.push(`${proj.px.toFixed(1)},${proj.py.toFixed(1)}`);
    }
    setHoverPatchPoints(pts.length === 4 ? pts.join(' ') : null);

    if (!drag.active) {
      const { clientX: cx, clientY: cy } = e;
      const { ra, dec } = hit.world;
      cursorHoverTimerRef.current = window.setTimeout(() => {
        setCursorTooltip({ x: cx, y: cy, ra, dec });
      }, CURSOR_TOOLTIP_DELAY_MS);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current.active) return;
    const wasMoved = dragRef.current.moved;
    dragRef.current.active = false;
    if (wasMoved || isBlinkerModalOpen || isInteractiveOverlayTarget(e.target)) return;

    const hit = screenToWorldCoords(e.clientX, e.clientY);
    if (!hit) return;

    const action = resolveGlobeClickAction(viewStateRef.current.fovDeg);
    if (action !== 'open_panel') {
      const nextFov = targetFovForClickAction(action);
      applyCameraState(hit.world.ra, hit.world.dec, nextFov, true);
      observatoryLogger.info('globe', 'Progressive click zoom', {
        stage: action,
        nextFovDeg: nextFov,
        ra: Number(hit.world.ra.toFixed(4)),
        dec: Number(hit.world.dec.toFixed(4)),
      });
      return;
    }

    const { raQ, decQ } = quantizeGridCenter(hit.world.ra, hit.world.dec, PANEL_GRID_STEP_DEG);
    const matchedBenchmark = BENCHMARK_TARGET_OPTIONS.find(
      (b) => angularSeparationDeg(raQ, decQ, b.ra, b.dec) <= 0.12,
    );
    observatoryLogger.info('globe', 'Opening sky panel in Blinker', {
      raQ,
      decQ,
      benchmarkSlug: matchedBenchmark?.slug ?? null,
    });
    if (matchedBenchmark) {
      openBlinkerModal?.(matchedBenchmark.slug);
    } else {
      openBlinkerModal?.({ ra: raQ, dec: decQ, fovArcmin: SPHEREX_PATCH_ARCMIN });
    }
  };

  useEffect(() => {
    isBlinkerModalOpenRef.current = isBlinkerModalOpen;
    const syncPauseState = () => {
      aladinThrottlerRef.current?.setPaused(isBlinkerModalOpen || document.hidden);
    };
    syncPauseState();
    document.addEventListener('visibilitychange', syncPauseState);
    return () => document.removeEventListener('visibilitychange', syncPauseState);
  }, [isBlinkerModalOpen]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    // Aladin's catalogCanvas calls stopPropagation() on wheel & mousedown; wake in capture phase
    // so wheel zoom and clicks over Aladin always resume 60 FPS rendering immediately.
    const wakeOnInputCapture = (e: Event) => {
      if (isBlinkerModalOpen || isInteractiveOverlayTarget(e.target)) return;
      aladinThrottlerRef.current?.wake();
    };
    const onWheel = (e: WheelEvent) => {
      if (isBlinkerModalOpen) return;
      aladinThrottlerRef.current?.wake();
      if (aladinReady && !aladinFailed) return;
      if (isInteractiveOverlayTarget(e.target)) return;
      e.preventDefault();
      const nextFov = clampGlobeFov(viewStateRef.current.fovDeg * (e.deltaY > 0 ? 1.18 : 0.84));
      applyCameraState(viewStateRef.current.ra, viewStateRef.current.dec, nextFov, false);
    };
    const clearDragState = () => {
      dragRef.current.active = false;
    };
    el.addEventListener('wheel', wakeOnInputCapture, { capture: true, passive: true });
    el.addEventListener('pointerdown', wakeOnInputCapture, { capture: true, passive: true });
    el.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('pointerup', clearDragState);
    window.addEventListener('pointercancel', clearDragState);
    return () => {
      el.removeEventListener('wheel', wakeOnInputCapture, { capture: true });
      el.removeEventListener('pointerdown', wakeOnInputCapture, { capture: true });
      el.removeEventListener('wheel', onWheel);
      window.removeEventListener('pointerup', clearDragState);
      window.removeEventListener('pointercancel', clearDragState);
    };
  }, [aladinFailed, aladinReady, applyCameraState, isBlinkerModalOpen]);

  const handleGlobeKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (isBlinkerModalOpen || e.target !== e.currentTarget) return;
    const action = resolveGlobeKeyboardAction({
      key: e.key,
      shiftKey: e.shiftKey,
      ctrlKey: e.ctrlKey,
      metaKey: e.metaKey,
      altKey: e.altKey,
    });
    if (!action) return;
    e.preventDefault();
    const cur = viewStateRef.current;
    if (action.type === 'pan') {
      const next = stepPanCoordinates(
        cur.ra,
        cur.dec,
        cur.fovDeg,
        action.dir,
        action.fine ? 0.35 : 1.0,
      );
      applyCameraState(next.ra, next.dec, cur.fovDeg, true);
      observatoryLogger.info('globe', 'Keyboard pan shortcut', {
        dir: action.dir,
        fine: action.fine,
        ...next,
      });
      return;
    }
    const nextFov = action.factor === 'reset' ? 70.0 : clampGlobeFov(cur.fovDeg * action.factor);
    applyCameraState(cur.ra, cur.dec, nextFov, true);
    observatoryLogger.info('globe', 'Keyboard zoom shortcut', {
      action: action.factor === 'reset' ? 'reset' : action.factor < 1 ? 'zoom_in' : 'zoom_out',
      fovDeg: Number(nextFov.toFixed(2)),
    });
  };

  const isPanelGridVisible = liveFrustum.fovDeg <= PANEL_GRID_FOV_DEG;

  return (
    <section className="globe-viewport-layer" aria-label="Interactive Night-Sky Observatory">
      <div
        ref={containerRef}
        data-testid="night-sky-globe"
        className="globe-viewport-layer"
        tabIndex={0}
        aria-label="Interactive Night-Sky Globe (Arrow keys or WASD to pan, + and - to zoom, 0 to reset)"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={() => {
          dragRef.current.active = false;
        }}
        onPointerLeave={() => {
          dragRef.current.active = false;
          clearCursorTooltipTimer();
          setCursorTooltip(null);
          setHoverPatchPoints(null);
        }}
        onKeyDown={handleGlobeKeyDown}
      >
        <div
          ref={aladinHostRef}
          className="aladin-host"
          style={{ display: aladinFailed ? 'none' : 'block' }}
        />

        {(!aladinReady || aladinFailed) && (
          <FallbackSkyCanvas
            ra={liveFrustum.ra}
            dec={liveFrustum.dec}
            fovDeg={liveFrustum.fovDeg}
            width={viewportSize.width}
            height={viewportSize.height}
            globalKnownObjects={globalKnownObjects}
            paused={isBlinkerModalOpen}
          />
        )}

        <GlobeOverlaySvg
          geometry={overlayGeometry}
          isPanelGridVisible={isPanelGridVisible}
          hoverPatchPoints={hoverPatchPoints}
        />

        {cursorTooltip && !isBlinkerModalOpen && (
          <div
            className="globe-cursor-tooltip mono"
            data-testid="globe-cursor-tooltip"
            style={{
              left: Math.min(viewportSize.width - 280, Math.max(12, cursorTooltip.x + 14)),
              top: Math.min(viewportSize.height - 44, Math.max(12, cursorTooltip.y + 14)),
            }}
          >
            {formatRaDecCompact(cursorTooltip.ra, cursorTooltip.dec)}
          </div>
        )}
      </div>

      <GlobeControlsDock
        showConstellations={showConstellations}
        projection={projection}
        fetchQualityMode={fetchQualityMode}
        onPan={(dir) => {
          const cur = viewStateRef.current;
          const next = stepPanCoordinates(cur.ra, cur.dec, cur.fovDeg, dir);
          applyCameraState(next.ra, next.dec, cur.fovDeg, true);
          observatoryLogger.info('globe', 'Accessible pan button pressed', { dir, ...next });
        }}
        onZoom={(factor) => {
          const cur = viewStateRef.current;
          const nextFov = factor === 'reset' ? 70.0 : clampGlobeFov(cur.fovDeg * factor);
          applyCameraState(cur.ra, cur.dec, nextFov, true);
          observatoryLogger.info('globe', 'Accessible zoom button pressed', {
            action: factor === 'reset' ? 'reset' : factor < 1 ? 'zoom_in' : 'zoom_out',
            fovDeg: Number(nextFov.toFixed(2)),
          });
        }}
        onToggleConstellations={() => {
          setShowConstellations((prev) => {
            const next = !prev;
            observatoryLogger.info('globe', 'Toggled constellation overlay', { visible: next });
            return next;
          });
        }}
        onChangeProjection={(nextProj) => {
          setProjection(nextProj);
          aladinInstanceRef.current?.setProjection?.(nextProj);
          observatoryLogger.info('globe', 'Projection changed', { projection: nextProj });
        }}
        onChangeQualityMode={(nextMode) => setFetchQualityMode?.(nextMode)}
        onOpenCenterPatch={() => {
          const cur = viewStateRef.current;
          const matchingPreset = BENCHMARK_TARGET_OPTIONS.find(
            (b) => angularSeparationDeg(cur.ra, cur.dec, b.ra, b.dec) <= 0.03,
          );
          if (matchingPreset) {
            openBlinkerModal?.(matchingPreset.slug);
          } else {
            const { raQ, decQ } = quantizeGridCenter(cur.ra, cur.dec, PANEL_GRID_STEP_DEG);
            openBlinkerModal?.({ ra: raQ, dec: decQ, fovArcmin: SPHEREX_PATCH_ARCMIN });
          }
        }}
      />

      <GlobeTelemetryBar
        ra={liveFrustum.ra}
        dec={liveFrustum.dec}
        fovDeg={liveFrustum.fovDeg}
        activeSurveyLabel={activeSurveyLabel}
        isPanelGridVisible={isPanelGridVisible}
        onCommitCoordinates={(nextRa, nextDec) => {
          applyCameraState(nextRa, nextDec, viewStateRef.current.fovDeg, true);
          observatoryLogger.info('globe', 'Navigated from inline telemetry coordinates', {
            ra: nextRa,
            dec: nextDec,
          });
        }}
      />
    </section>
  );
};
