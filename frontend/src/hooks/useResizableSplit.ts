import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Read a bounded finite number from `window.localStorage`, returning `fallback`
 * when SSR-rendered, missing, unparseable, or outside `[min, max]`.
 *
 * @param key - LocalStorage key.
 * @param fallback - Default value when no valid stored number exists.
 * @param min - Inclusive minimum valid bound.
 * @param max - Inclusive maximum valid bound.
 * @returns Validated stored number or `fallback`.
 */
export function readStoredNumber(key: string, fallback: number, min: number, max: number): number {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null || raw.trim() === '') return fallback;
    const val = Number(raw);
    return Number.isFinite(val) && val >= min && val <= max ? val : fallback;
  } catch {
    return fallback;
  }
}

/** Storage keys and geometry bounds configuring {@link useResizableSplit}. */
export interface UseResizableSplitOptions {
  collapsedStorageKey?: string;
  widthStorageKey?: string;
  splitRatioStorageKey?: string;
  defaultWidth?: number;
  minWidth?: number;
  maxWidth?: number;
  defaultSplitRatio?: number;
  minSplitRatio?: number;
  maxSplitRatio?: number;
}

/**
 * Manage Blinker sidebar collapse state, horizontal sidebar width dragging,
 * vertical controls/known-objects split dragging, and accordion fold toggles
 * with persistent `localStorage` state.
 *
 * @param options - Optional storage keys and clamp bounds.
 * @returns Current layout dimensions, fold states, toggle callbacks, and pointer drag starters.
 */
export function useResizableSplit(options: UseResizableSplitOptions = {}) {
  const {
    collapsedStorageKey = 'sh_blinker_sidebar_collapsed_v1',
    widthStorageKey = 'sh_blinker_sidebar_width_v1',
    splitRatioStorageKey = 'sh_blinker_split_ratio_v1',
    defaultWidth = 292,
    minWidth = 240,
    maxWidth = 420,
    defaultSplitRatio = 0.56,
    minSplitRatio = 0.25,
    maxSplitRatio = 0.75,
  } = options;

  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return window.localStorage.getItem(collapsedStorageKey) === 'true';
    } catch {
      return false;
    }
  });
  const [sidebarWidth, setSidebarWidth] = useState<number>(() =>
    readStoredNumber(widthStorageKey, defaultWidth, minWidth, maxWidth),
  );
  const [splitRatio, setSplitRatio] = useState<number>(() =>
    readStoredNumber(splitRatioStorageKey, defaultSplitRatio, minSplitRatio, maxSplitRatio),
  );
  const [recentsFolded, setRecentsFolded] = useState<boolean>(true);
  const [controlsFolded, setControlsFolded] = useState<boolean>(false);
  const [objectsFolded, setObjectsFolded] = useState<boolean>(false);
  const dragCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => () => dragCleanupRef.current?.(), []);

  const toggleSidebarCollapsed = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(collapsedStorageKey, String(next));
      } catch {
        // Ignore storage errors
      }
      return next;
    });
  }, [collapsedStorageKey]);

  const toggleRecentsFold = useCallback(() => setRecentsFolded((prev) => !prev), []);
  const toggleControlsFold = useCallback(() => setControlsFolded((prev) => !prev), []);
  const toggleObjectsFold = useCallback(() => setObjectsFolded((prev) => !prev), []);

  const startSidebarWidthResize = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      dragCleanupRef.current?.();
      const startX = e.clientX;
      const startW = sidebarWidth;
      let latestW = startW;
      const onMove = (ev: PointerEvent) => {
        latestW = Math.max(
          minWidth,
          Math.min(maxWidth, Math.round(startW + (ev.clientX - startX))),
        );
        setSidebarWidth(latestW);
      };
      const cleanup = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onEnd);
        window.removeEventListener('pointercancel', onEnd);
        dragCleanupRef.current = null;
      };
      const onEnd = () => {
        cleanup();
        try {
          window.localStorage.setItem(widthStorageKey, String(latestW));
        } catch {
          // Ignore storage errors
        }
      };
      dragCleanupRef.current = cleanup;
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onEnd);
      window.addEventListener('pointercancel', onEnd);
    },
    [maxWidth, minWidth, sidebarWidth, widthStorageKey],
  );

  const startSplitRatioResize = useCallback(
    (e: React.PointerEvent<HTMLDivElement>, sidebarEl: HTMLElement | null) => {
      e.preventDefault();
      e.stopPropagation();
      if (!sidebarEl) return;
      dragCleanupRef.current?.();
      const totalH = Math.max(160, sidebarEl.getBoundingClientRect().height);
      const startY = e.clientY;
      const startRatio = splitRatio;
      let latestRatio = startRatio;
      const onMove = (ev: PointerEvent) => {
        latestRatio = Math.max(
          minSplitRatio,
          Math.min(maxSplitRatio, startRatio + (ev.clientY - startY) / totalH),
        );
        setSplitRatio(latestRatio);
      };
      const cleanup = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onEnd);
        window.removeEventListener('pointercancel', onEnd);
        dragCleanupRef.current = null;
      };
      const onEnd = () => {
        cleanup();
        try {
          window.localStorage.setItem(splitRatioStorageKey, latestRatio.toFixed(3));
        } catch {
          // Ignore storage errors
        }
      };
      dragCleanupRef.current = cleanup;
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onEnd);
      window.addEventListener('pointercancel', onEnd);
    },
    [maxSplitRatio, minSplitRatio, splitRatio, splitRatioStorageKey],
  );

  return {
    sidebarCollapsed,
    sidebarWidth,
    splitRatio,
    recentsFolded,
    controlsFolded,
    objectsFolded,
    toggleSidebarCollapsed,
    toggleRecentsFold,
    toggleControlsFold,
    toggleObjectsFold,
    startSidebarWidthResize,
    startSplitRatioResize,
  };
}
