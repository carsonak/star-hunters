/**
 * Adaptive network quality estimation and fetch-precision policy (`float32` vs `float16`).
 *
 * Kept React-free so Node 24's `node --test` executes this module directly.
 */

import type { BinaryDtype, FetchQualityMode } from '../types/contracts.ts';

/** Threshold below which `auto` mode drops from `float32` to `float16`. */
export const SLOW_NETWORK_DROP_MBPS = 3.0;

/** Recovery threshold (`>= 5 Mbps`) to switch back to `float32` (hysteresis). */
export const FAST_NETWORK_RECOVERY_MBPS = 5.0;

/** Stall timeout in milliseconds before `auto` aborts a stalled `float32` frame and retries `float16`. */
export const AUTO_FLOAT32_STALL_MS = 4500;

export const QUALITY_MODE_STORAGE_KEY = 'sh_fetch_quality_mode_v1';

interface NavigatorConnectionLike {
  downlink?: number;
  effectiveType?: string;
  saveData?: boolean;
}

export interface NetworkQualitySnapshot {
  isSlow: boolean;
  measuredMbps: number | null;
  reason: 'navigator' | 'throughput' | 'stall' | 'nominal';
}

/**
 * Read persisted {@link FetchQualityMode} from `localStorage`, defaulting to `'auto'`.
 */
export function readStoredQualityMode(): FetchQualityMode {
  if (typeof window === 'undefined') return 'auto';
  try {
    const raw = window.localStorage.getItem(QUALITY_MODE_STORAGE_KEY);
    if (raw === 'auto' || raw === 'full' || raw === 'compressed') {
      return raw;
    }
  } catch {
    // Ignore storage errors
  }
  return 'auto';
}

/**
 * Persist {@link FetchQualityMode} to `localStorage`.
 */
export function saveQualityMode(mode: FetchQualityMode): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(QUALITY_MODE_STORAGE_KEY, mode);
  } catch {
    // Ignore storage errors
  }
}

/**
 * Inspect `navigator.connection` (Network Information API) when available.
 */
export function readNavigatorConnectionHint(conn?: NavigatorConnectionLike | null): {
  isSlow: boolean;
  downlinkMbps: number | null;
} {
  const navConn =
    conn ??
    (typeof navigator !== 'undefined'
      ? ((navigator as Navigator & { connection?: NavigatorConnectionLike }).connection ?? null)
      : null);
  if (!navConn) {
    return { isSlow: false, downlinkMbps: null };
  }
  if (navConn.saveData) {
    return { isSlow: true, downlinkMbps: navConn.downlink ?? null };
  }
  const eff = (navConn.effectiveType || '').toLowerCase();
  if (eff === 'slow-2g' || eff === '2g' || eff === '3g') {
    return {
      isSlow: true,
      downlinkMbps: typeof navConn.downlink === 'number' ? navConn.downlink : null,
    };
  }
  if (typeof navConn.downlink === 'number' && Number.isFinite(navConn.downlink)) {
    return {
      isSlow: navConn.downlink > 0 && navConn.downlink < SLOW_NETWORK_DROP_MBPS,
      downlinkMbps: navConn.downlink,
    };
  }
  return { isSlow: false, downlinkMbps: null };
}

/**
 * Rolling EWMA network throughput tracker with hysteresis (`< 3 Mbps` drop,
 * `>= 5 Mbps` recovery) and stall detection.
 */
export class NetworkQualityTracker {
  private ewmaMbps: number | null = null;
  private isSlow = false;
  private reason: NetworkQualitySnapshot['reason'] = 'nominal';
  private readonly getNavHint: () => { isSlow: boolean; downlinkMbps: number | null };

  constructor(
    getNavHint: () => { isSlow: boolean; downlinkMbps: number | null } = () =>
      readNavigatorConnectionHint(),
  ) {
    this.getNavHint = getNavHint;
    const hint = this.getNavHint();
    if (hint.downlinkMbps !== null && hint.downlinkMbps > 0) {
      this.ewmaMbps = hint.downlinkMbps;
    }
    if (hint.isSlow) {
      this.isSlow = true;
      this.reason = 'navigator';
    }
  }

  /** Record a completed network binary transfer sample (ignores tiny/instant cache reads). */
  public recordTransfer(bytes: number, elapsedMs: number): NetworkQualitySnapshot {
    if (!Number.isFinite(bytes) || !Number.isFinite(elapsedMs) || bytes < 32_768 || elapsedMs < 5) {
      return this.getSnapshot();
    }
    const sampleMbps = (bytes * 8) / (elapsedMs / 1000) / 1_000_000;
    this.ewmaMbps = this.ewmaMbps === null ? sampleMbps : 0.35 * sampleMbps + 0.65 * this.ewmaMbps;

    if (this.ewmaMbps < SLOW_NETWORK_DROP_MBPS) {
      this.isSlow = true;
      this.reason = 'throughput';
    } else if (this.ewmaMbps >= FAST_NETWORK_RECOVERY_MBPS) {
      const hint = this.getNavHint();
      if (!hint.isSlow) {
        this.isSlow = false;
        this.reason = 'nominal';
      }
    }
    return this.getSnapshot();
  }

  /** Record a stalled or failed `float32` transfer, immediately dropping to slow mode. */
  public recordStall(): NetworkQualitySnapshot {
    this.isSlow = true;
    this.reason = 'stall';
    this.ewmaMbps = this.ewmaMbps === null ? 1.5 : Math.min(this.ewmaMbps, 1.5);
    return this.getSnapshot();
  }

  /** Return current slow-network state, preferring live EWMA and falling back to navigator hints. */
  public getSnapshot(): NetworkQualitySnapshot {
    const hint = this.getNavHint();
    if (this.ewmaMbps === null && hint.isSlow) {
      return { isSlow: true, measuredMbps: hint.downlinkMbps, reason: 'navigator' };
    }
    if (hint.isSlow && !this.isSlow) {
      return {
        isSlow: true,
        measuredMbps: this.ewmaMbps ?? hint.downlinkMbps,
        reason: 'navigator',
      };
    }
    return {
      isSlow: this.isSlow,
      measuredMbps: this.ewmaMbps !== null ? Number(this.ewmaMbps.toFixed(2)) : hint.downlinkMbps,
      reason: this.reason,
    };
  }

  /** Resolve the target wire dtype for the given {@link FetchQualityMode}. */
  public resolveTargetDtype(mode: FetchQualityMode): BinaryDtype {
    if (mode === 'full') return 'float32';
    if (mode === 'compressed') return 'float16';
    return this.getSnapshot().isSlow ? 'float16' : 'float32';
  }
}

export const defaultNetworkQualityTracker = new NetworkQualityTracker();
