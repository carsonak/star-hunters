import { useCallback, useState } from 'react';
import type { RecentPatchEntry } from '../types/contracts';

export const RECENT_PATCHES_STORAGE_KEY = 'sh_recent_patches_v1';
export const MAX_RECENT_PATCHES = 12;

/**
 * Insert or promote `entry` to the front of `list`, deduplicating by `id` and capping length.
 */
export function upsertRecentPatch(
  list: ReadonlyArray<RecentPatchEntry>,
  entry: RecentPatchEntry,
  maxItems = MAX_RECENT_PATCHES,
): RecentPatchEntry[] {
  const next = [entry, ...list.filter((item) => item.id !== entry.id)];
  return next.slice(0, Math.max(1, maxItems));
}

function readStoredRecentPatches(): RecentPatchEntry[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(RECENT_PATCHES_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (item): item is RecentPatchEntry =>
          Boolean(item) &&
          typeof item.id === 'string' &&
          typeof item.ra === 'number' &&
          typeof item.dec === 'number' &&
          typeof item.title === 'string',
      )
      .slice(0, MAX_RECENT_PATCHES);
  } catch {
    return [];
  }
}

function writeStoredRecentPatches(entries: ReadonlyArray<RecentPatchEntry>): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(RECENT_PATCHES_STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Ignore storage quota errors
  }
}

/**
 * Manage recently opened Blinker sky patches persisted in `localStorage`.
 */
export function useRecentPatches() {
  const [recentPatches, setRecentPatches] = useState<RecentPatchEntry[]>(() =>
    readStoredRecentPatches(),
  );

  const recordRecentPatch = useCallback((entry: RecentPatchEntry) => {
    setRecentPatches((prev) => {
      const next = upsertRecentPatch(prev, entry);
      writeStoredRecentPatches(next);
      return next;
    });
  }, []);

  const clearRecentPatches = useCallback(() => {
    setRecentPatches([]);
    writeStoredRecentPatches([]);
  }, []);

  return {
    recentPatches,
    recordRecentPatch,
    clearRecentPatches,
  };
}
