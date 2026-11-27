/**
 * 3G-Optimised Two-Tier Client Cache (In-Memory LRU Map + Browser IndexedDB).
 *
 * Ensures that once a sky patch's raw binary frames, manifest, and catalog overlays
 * are downloaded, all user-configurable options (combining epochs, wavelength filtering,
 * checking/unchecking known objects, pan/zoom, and click-to-track) execute locally
 * without refetching.
 *
 * Tier 1 — Insertion-ordered in-memory `Map`s acting as O(1) LRU caches so hot
 *          UI interactions resolve synchronously in well under 5 ms.
 * Tier 2 — IndexedDB database `star_hunters_patch_cache_v1` with the `sky_patches`
 *          and `catalogs` object stores so typed-array buffers survive page reloads.
 *
 * In headless/Node test environments `indexedDB` is absent, so every operation
 * gracefully degrades to the in-memory tier instead of throwing.
 */

import type {
  BinaryDtype,
  CatalogsResponse,
  SkyPatchManifest,
  SpectrumPayload,
} from '../types/contracts.ts';

/** Cached multi-epoch binary bundle and metadata persisted per sky patch. */
export interface CachedSkyPatchBundle {
  cacheKey: string;
  dtype?: BinaryDtype;
  manifest: SkyPatchManifest;
  epochBuffers: Float32Array[][];
  flagsBuffers: Uint8Array[];
  spectrum?: SpectrumPayload | null;
  storedAt: number;
  lastAccessed: number;
}

/** A loaded sky patch bundle before it receives cache bookkeeping fields. */
export type SkyPatchBundleInput = Omit<
  CachedSkyPatchBundle,
  'cacheKey' | 'storedAt' | 'lastAccessed'
> &
  Partial<Pick<CachedSkyPatchBundle, 'storedAt' | 'lastAccessed'>>;

/** Diagnostic statistics snapshot returned by {@link ClientPatchCache.getStats}. */
export interface ClientPatchCacheStats {
  memoryPatches: number;
  memoryCatalogs: number;
  maxEntries: number;
  indexedDbAvailable: boolean;
  patchHits: number;
  patchMisses: number;
  catalogHits: number;
  catalogMisses: number;
}

/** Timestamped entry constraint required by {@link TieredLruStore}. */
export interface LruTimestampedEntry {
  storedAt: number;
  lastAccessed: number;
}

interface CatalogCacheEntry extends LruTimestampedEntry {
  value: CatalogsResponse;
}

/**
 * Canonical `preset:{slug}` cache key used by the preset tour.
 *
 * @param slug - Preset target identifier.
 * @returns Canonical preset cache key.
 */
export function presetCacheKey(slug: string): string {
  return `preset:${slug.trim()}`;
}

/**
 * Canonical cache key for a live `ra/dec/fov` query. The backend `grid_key`
 * (see `computeGridKey`) is already stable and URL-safe, so it is used verbatim.
 *
 * @param gridKey - Quantized backend grid key.
 * @returns Normalized cache key.
 */
export function gridCacheKey(gridKey: string): string {
  return gridKey.trim();
}

/**
 * Precision-qualified cache key so `float32` and `float16` bundles for the same
 * sky patch are stored as distinct entries in RAM and IndexedDB.
 */
export function livePatchCacheKey(gridKey: string, dtype: BinaryDtype): string {
  return `${gridKey.trim()}:${dtype}`;
}

/** Default 7-day client cache TTL matching backend IRSA weekly releases. */
export const DEFAULT_CLIENT_PATCH_TTL_MS = 7 * 24 * 3600 * 1000;

/** IndexedDB database name for persisted sky patch and catalog caches. */
export const PATCH_CACHE_DB_NAME = 'star_hunters_patch_cache_v1';
/** IndexedDB object store name for decoded sky patch bundles. */
export const PATCH_CACHE_STORE_PATCHES = 'sky_patches';
/** IndexedDB object store name for Stage A catalog overlay responses. */
export const PATCH_CACHE_STORE_CATALOGS = 'catalogs';
/** Default per-store LRU capacity in entries. */
export const DEFAULT_MAX_MEMORY_PATCHES = 16;

/**
 * Generic two-tier LRU store backed by an in-memory `Map` (Tier 1) and an
 * IndexedDB object store (Tier 2).
 */
export class TieredLruStore<T extends LruTimestampedEntry> {
  private memory = new Map<string, T>();
  /** Session-local LRU stamps, used to prune IndexedDB even before access writes flush. */
  private accessOrder = new Map<string, number>();
  private hits = 0;
  private misses = 0;
  private readonly storeName: string;
  private readonly maxEntries: number;
  private readonly ttlMs: number | null;
  private readonly nextAccessStamp: () => number;
  private readonly openDb: () => Promise<IDBDatabase | null>;

  /**
   * Create a two-tier LRU store bound to an IndexedDB object store.
   *
   * @param options - Store name, capacity, monotonic stamp generator, and DB opener.
   */
  constructor(options: {
    storeName: string;
    maxEntries: number;
    ttlMs?: number | null;
    nextAccessStamp: () => number;
    openDb: () => Promise<IDBDatabase | null>;
  }) {
    this.storeName = options.storeName;
    this.maxEntries = options.maxEntries;
    this.ttlMs = options.ttlMs ?? null;
    this.nextAccessStamp = options.nextAccessStamp;
    this.openDb = options.openDb;
  }

  /** Number of entries currently resident in the Tier 1 in-memory `Map`. */
  public get memorySize(): number {
    return this.memory.size;
  }

  /** Cumulative cache hit count across Tier 1 and Tier 2 lookups. */
  public get hitCount(): number {
    return this.hits;
  }

  /** Cumulative cache miss count across Tier 1 and Tier 2 lookups. */
  public get missCount(): number {
    return this.misses;
  }

  /**
   * Retrieve an entry from Tier 1 memory or fall back to Tier 2 IndexedDB.
   *
   * @param key - Canonical store key.
   * @returns Cached entry if present, otherwise `null`.
   */
  private isExpired(entry: T): boolean {
    if (
      this.ttlMs === null ||
      this.ttlMs <= 0 ||
      !entry.storedAt ||
      entry.storedAt < 1_000_000_000_000
    ) {
      return false;
    }
    return Date.now() - entry.storedAt > this.ttlMs;
  }

  public async get(key: string): Promise<T | null> {
    const inMem = this.memory.get(key);
    if (inMem) {
      if (this.isExpired(inMem)) {
        await this.delete(key);
        this.misses += 1;
        return null;
      }
      this.hits += 1;
      // Hot path: touch memory only. Re-serializing a multi-megabyte bundle to
      // IndexedDB on every read would defeat the <5 ms requirement; the
      // session access order is enough to rank persistent LRU pruning.
      this.touchMemory(key, inMem);
      return inMem;
    }

    const fromIdb = await this.readFromIndexedDb(key);
    if (fromIdb) {
      if (this.isExpired(fromIdb)) {
        await this.delete(key);
        this.misses += 1;
        return null;
      }
      this.hits += 1;
      const stamp = this.nextAccessStamp();
      fromIdb.lastAccessed = stamp;
      this.accessOrder.set(key, stamp);
      this.putMemory(key, fromIdb);
      return fromIdb;
    }

    this.misses += 1;
    return null;
  }

  /**
   * Synchronously retrieve an entry from Tier 1 memory without touching IndexedDB.
   *
   * @param key - Canonical store key.
   * @returns Cached entry in memory, or `null` on a Tier 1 miss.
   */
  public getSync(key: string): T | null {
    const inMem = this.memory.get(key);
    if (!inMem) {
      this.misses += 1;
      return null;
    }
    this.hits += 1;
    this.touchMemory(key, inMem);
    return inMem;
  }

  /**
   * Check whether an entry exists in either Tier 1 memory or Tier 2 IndexedDB.
   *
   * @param key - Canonical store key.
   * @returns `true` if present in either tier.
   */
  public async has(key: string): Promise<boolean> {
    if (this.memory.has(key)) return true;
    return (await this.readFromIndexedDb(key)) !== null;
  }

  /**
   * Store an entry in both Tier 1 memory and Tier 2 IndexedDB, enforcing LRU bounds.
   *
   * @param key - Canonical store key.
   * @param value - Entry to persist.
   */
  public async put(key: string, value: T): Promise<void> {
    const stamp = this.nextAccessStamp();
    const entry: T = {
      ...value,
      storedAt: value.storedAt || Date.now(),
      lastAccessed: stamp,
    };
    this.accessOrder.set(key, stamp);
    this.putMemory(key, entry);
    await this.writeToIndexedDb(key, entry);
    await this.pruneIndexedDb();
  }

  /**
   * Remove an entry from both Tier 1 memory and Tier 2 IndexedDB.
   *
   * @param key - Canonical store key.
   */
  public async delete(key: string): Promise<void> {
    this.memory.delete(key);
    this.accessOrder.delete(key);
    await this.deleteFromIndexedDb(key);
  }

  /** Evict all Tier 1 in-memory entries while leaving Tier 2 IndexedDB intact. */
  public clearMemory(): void {
    this.memory.clear();
    this.accessOrder.clear();
  }

  private touchMemory(key: string, entry: T): number {
    const stamp = this.nextAccessStamp();
    entry.lastAccessed = stamp;
    this.memory.delete(key);
    this.memory.set(key, entry);
    this.accessOrder.set(key, stamp);
    return stamp;
  }

  private putMemory(key: string, entry: T): void {
    this.memory.delete(key);
    this.memory.set(key, entry);
    while (this.memory.size > this.maxEntries) {
      const oldestKey = this.memory.keys().next().value;
      if (oldestKey === undefined) break;
      this.memory.delete(oldestKey);
      this.accessOrder.delete(oldestKey);
    }
  }

  private async readFromIndexedDb(key: string): Promise<T | null> {
    const db = await this.openDb();
    if (!db) return null;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(this.storeName, 'readonly');
        const store = tx.objectStore(this.storeName);
        const req = store.get(key);
        req.onsuccess = () => resolve((req.result as T) ?? null);
        req.onerror = () => resolve(null);
        tx.onabort = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  private async writeToIndexedDb(key: string, value: T): Promise<void> {
    const db = await this.openDb();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(this.storeName, 'readwrite');
        tx.objectStore(this.storeName).put(value, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  private async deleteFromIndexedDb(key: string): Promise<void> {
    const db = await this.openDb();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(this.storeName, 'readwrite');
        tx.objectStore(this.storeName).delete(key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  /**
   * Bound the persistent store to `maxEntries`, deleting the least-recently-used
   * rows first. Session access stamps take precedence over persisted timestamps
   * so an access performed earlier in this page session is never mis-ranked just
   * because its IndexedDB write has not flushed yet.
   */
  private async pruneIndexedDb(): Promise<void> {
    const db = await this.openDb();
    if (!db) return;

    await new Promise<void>((resolve) => {
      try {
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        const rows: Array<{ key: IDBValidKey; stamp: number }> = [];
        const cursorReq = store.openCursor();

        cursorReq.onsuccess = () => {
          const cursor = cursorReq.result;
          if (!cursor) {
            if (rows.length > this.maxEntries) {
              rows.sort((a, b) => a.stamp - b.stamp);
              const excess = rows.length - this.maxEntries;
              for (let i = 0; i < excess; i += 1) {
                store.delete(rows[i].key);
                this.accessOrder.delete(String(rows[i].key));
              }
            }
            return;
          }

          const keyStr = String(cursor.key);
          const value = cursor.value as Partial<LruTimestampedEntry> | null;
          const stamp = this.accessOrder.get(keyStr) ?? value?.lastAccessed ?? value?.storedAt ?? 0;
          rows.push({ key: cursor.key, stamp });
          cursor.continue();
        };
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }
}

/**
 * Two-tier client cache managing sky patch binary bundles and catalog overlays
 * across an in-memory LRU tier and persistent browser IndexedDB storage.
 */
export class ClientPatchCache {
  private readonly maxEntries: number;
  private accessClock = 0;
  private reportedPatchHits = 0;
  private reportedCatalogHits = 0;
  private dbPromise: Promise<IDBDatabase | null> | null = null;
  private readonly patchStore: TieredLruStore<CachedSkyPatchBundle>;
  private readonly catalogStore: TieredLruStore<CatalogCacheEntry>;

  /**
   * Construct a {@link ClientPatchCache} with a shared per-store LRU capacity.
   *
   * @param maxEntries - Maximum number of entries retained per store.
   */
  constructor(
    maxEntries: number = DEFAULT_MAX_MEMORY_PATCHES,
    ttlMs: number = DEFAULT_CLIENT_PATCH_TTL_MS,
  ) {
    this.maxEntries = Number.isFinite(maxEntries)
      ? Math.max(0, Math.floor(maxEntries))
      : DEFAULT_MAX_MEMORY_PATCHES;

    const nextAccessStamp = () => this.nextAccessStamp();
    const openDb = () => this.openDb();

    this.patchStore = new TieredLruStore<CachedSkyPatchBundle>({
      storeName: PATCH_CACHE_STORE_PATCHES,
      maxEntries: this.maxEntries,
      ttlMs,
      nextAccessStamp,
      openDb,
    });
    this.catalogStore = new TieredLruStore<CatalogCacheEntry>({
      storeName: PATCH_CACHE_STORE_CATALOGS,
      maxEntries: this.maxEntries,
      ttlMs,
      nextAccessStamp,
      openDb,
    });
  }

  // -------------------------------------------------------------------------
  // Sky patch bundle — Tier 1 + Tier 2
  // -------------------------------------------------------------------------

  /**
   * Retrieve a cached sky patch bundle from memory or IndexedDB.
   *
   * @param cacheKey - Canonical preset or grid key.
   * @returns Cached bundle if present, otherwise `null`.
   */
  public getPatch(cacheKey: string): Promise<CachedSkyPatchBundle | null> {
    return this.patchStore.get(cacheKey);
  }

  /**
   * Memory-only synchronous lookup (no promise, no IndexedDB). Useful for
   * 60 FPS render loops that must never await storage.
   *
   * @param cacheKey - Canonical preset or grid key.
   * @returns Cached bundle from Tier 1 memory, or `null`.
   */
  public getPatchSync(cacheKey: string): CachedSkyPatchBundle | null {
    return this.patchStore.getSync(cacheKey);
  }

  /**
   * Check whether a sky patch bundle exists in memory or IndexedDB.
   *
   * @param cacheKey - Canonical preset or grid key.
   * @returns `true` if cached in either tier.
   */
  public hasPatch(cacheKey: string): Promise<boolean> {
    return this.patchStore.has(cacheKey);
  }

  /**
   * Persist a sky patch bundle in both Tier 1 memory and Tier 2 IndexedDB.
   *
   * @param bundle - Decoded patch bundle to store.
   */
  public putPatch(bundle: CachedSkyPatchBundle): Promise<void> {
    return this.patchStore.put(bundle.cacheKey, bundle);
  }

  /**
   * Delete a sky patch bundle from both memory and IndexedDB.
   *
   * @param cacheKey - Canonical preset or grid key.
   */
  public deletePatch(cacheKey: string): Promise<void> {
    return this.patchStore.delete(cacheKey);
  }

  /**
   * Return the cached patch, or invoke `loader` exactly once on a miss and cache
   * its result. A warm cache therefore performs **zero network requests**.
   *
   * @param cacheKey - Canonical preset or grid key.
   * @param loader - Async loader invoked only on a cache miss.
   * @returns Cached or newly loaded bundle, or `null` if `loader` returned `null`.
   */
  public async getOrFetchPatch(
    cacheKey: string,
    loader: () => Promise<SkyPatchBundleInput | null>,
  ): Promise<CachedSkyPatchBundle | null> {
    const cached = await this.getPatch(cacheKey);
    if (cached) return cached;

    const loaded = await loader();
    if (!loaded) return null;

    const entry: CachedSkyPatchBundle = {
      ...loaded,
      cacheKey,
      storedAt: loaded.storedAt || Date.now(),
      lastAccessed: this.nextAccessStamp(),
    };
    await this.putPatch(entry);
    return entry;
  }

  // -------------------------------------------------------------------------
  // Catalog overlays — Tier 1 + Tier 2
  // -------------------------------------------------------------------------

  /**
   * Retrieve cached catalog overlays for a quantized `gridKey`.
   *
   * @param gridKey - Canonical quantized grid key.
   * @returns Cached {@link CatalogsResponse} if present, otherwise `null`.
   */
  public async getCatalogs(gridKey: string): Promise<CatalogsResponse | null> {
    const entry = await this.catalogStore.get(gridKey);
    return entry ? entry.value : null;
  }

  /**
   * Check whether catalog overlays for `gridKey` exist in memory or IndexedDB.
   *
   * @param gridKey - Canonical quantized grid key.
   * @returns `true` if cached in either tier.
   */
  public hasCatalogs(gridKey: string): Promise<boolean> {
    return this.catalogStore.has(gridKey);
  }

  /**
   * Persist catalog overlays for `gridKey` in both memory and IndexedDB.
   *
   * @param gridKey - Canonical quantized grid key.
   * @param catalogs - Stage A catalog response payload.
   */
  public putCatalogs(gridKey: string, catalogs: CatalogsResponse): Promise<void> {
    const now = Date.now();
    return this.catalogStore.put(gridKey, {
      value: catalogs,
      storedAt: now,
      lastAccessed: now,
    });
  }

  /**
   * Delete cached catalog overlays for `gridKey` from memory and IndexedDB.
   *
   * @param gridKey - Canonical quantized grid key.
   */
  public deleteCatalogs(gridKey: string): Promise<void> {
    return this.catalogStore.delete(gridKey);
  }

  /**
   * Return cached catalogs, or load and cache them once on a miss.
   *
   * @param gridKey - Canonical quantized grid key.
   * @param loader - Async loader invoked only on a cache miss.
   * @returns Cached or newly loaded {@link CatalogsResponse}, or `null`.
   */
  public async getOrFetchCatalogs(
    gridKey: string,
    loader: () => Promise<CatalogsResponse | null>,
  ): Promise<CatalogsResponse | null> {
    const cached = await this.getCatalogs(gridKey);
    if (cached) return cached;

    const loaded = await loader();
    if (!loaded) return null;
    await this.putCatalogs(gridKey, loaded);
    return loaded;
  }

  // -------------------------------------------------------------------------
  // Housekeeping & diagnostics
  // -------------------------------------------------------------------------

  /** Clear all Tier 1 in-memory patch and catalog caches. */
  public clearMemory(): void {
    this.patchStore.clearMemory();
    this.catalogStore.clearMemory();
  }

  /** Clear both tiers (memory + persistent IndexedDB stores). */
  public async clearAll(): Promise<void> {
    this.clearMemory();
    const db = await this.openDb();
    if (!db) return;
    await new Promise<void>((resolve) => {
      try {
        const tx = db.transaction(
          [PATCH_CACHE_STORE_PATCHES, PATCH_CACHE_STORE_CATALOGS],
          'readwrite',
        );
        tx.objectStore(PATCH_CACHE_STORE_PATCHES).clear();
        tx.objectStore(PATCH_CACHE_STORE_CATALOGS).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  /** Current entry counts resident in Tier 1 memory. */
  public get size(): { patches: number; catalogs: number } {
    return {
      patches: this.patchStore.memorySize,
      catalogs: this.catalogStore.memorySize,
    };
  }

  /** Maximum number of entries retained per store before LRU eviction. */
  public get capacity(): number {
    return this.maxEntries;
  }

  /**
   * Return hit/miss counters and Tier 1 occupancy statistics.
   *
   * @returns Snapshot of cache occupancy and hit/miss counters.
   */
  public getStats(): ClientPatchCacheStats {
    return {
      memoryPatches: this.patchStore.memorySize,
      memoryCatalogs: this.catalogStore.memorySize,
      maxEntries: this.maxEntries,
      indexedDbAvailable: this.isIndexedDbAvailable(),
      patchHits: this.patchStore.hitCount,
      patchMisses: this.patchStore.missCount,
      catalogHits: this.catalogStore.hitCount,
      catalogMisses: this.catalogStore.missCount,
    };
  }

  /**
   * Consume unsent local cache hit deltas for piggybacked `X-Cache-Delta` telemetry.
   *
   * Returns `null` when both deltas are zero so cold requests emit no extra headers.
   */
  public consumeTelemetryDeltaHeader(): string | null {
    const pDelta = Math.max(0, this.patchStore.hitCount - this.reportedPatchHits);
    const cDelta = Math.max(0, this.catalogStore.hitCount - this.reportedCatalogHits);
    this.reportedPatchHits = this.patchStore.hitCount;
    this.reportedCatalogHits = this.catalogStore.hitCount;
    if (pDelta === 0 && cDelta === 0) {
      return null;
    }
    return `p=${pDelta},c=${cDelta}`;
  }

  /**
   * True only when the browser exposes a usable IndexedDB implementation.
   *
   * @returns Whether persistent IndexedDB storage is available.
   */
  public isPersistentStorageAvailable(): boolean {
    return this.isIndexedDbAvailable();
  }

  private nextAccessStamp(): number {
    this.accessClock = Math.max(Date.now(), this.accessClock + 1);
    return this.accessClock;
  }

  private isIndexedDbAvailable(): boolean {
    return (
      typeof globalThis !== 'undefined' &&
      'indexedDB' in globalThis &&
      globalThis.indexedDB !== null &&
      globalThis.indexedDB !== undefined
    );
  }

  private openDb(): Promise<IDBDatabase | null> {
    if (!this.isIndexedDbAvailable()) {
      return Promise.resolve(null);
    }
    if (this.dbPromise) {
      return this.dbPromise;
    }
    this.dbPromise = new Promise((resolve) => {
      try {
        const req = globalThis.indexedDB.open(PATCH_CACHE_DB_NAME, 1);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(PATCH_CACHE_STORE_PATCHES)) {
            db.createObjectStore(PATCH_CACHE_STORE_PATCHES);
          }
          if (!db.objectStoreNames.contains(PATCH_CACHE_STORE_CATALOGS)) {
            db.createObjectStore(PATCH_CACHE_STORE_CATALOGS);
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
    return this.dbPromise;
  }
}

/** Shared singleton client patch cache instance used by the observatory UI. */
export const defaultClientPatchCache = new ClientPatchCache();
