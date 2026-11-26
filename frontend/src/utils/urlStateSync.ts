import type { ComparisonMode } from '../types/contracts';

/**
 * ISS-14: Shareable URL query-state synchronization for the Aladin wide-field
 * sky map. The helpers in this module are intentionally DOM-optional so they
 * can be unit-tested in Node and reused by the React app without reloading the
 * page (`history.replaceState` only).
 */

export const SUPPORTED_COMPARISON_MODES: readonly ComparisonMode[] = [
  'blink',
  'subtract',
  'coadd',
] as const;

export const URL_STATE_DEFAULTS = {
  fov: 5.0,
  mode: 'blink' as ComparisonMode,
  wlMin: 0.75,
  wlMax: 5.0,
} as const;

/**
 * Normalized representation of the sky-view query parameters.
 *
 * `ra`/`dec` stay `null` when absent so the app can distinguish "no coordinate
 * supplied" from an intentional 0h/0deg target. The remaining fields fall back
 * to their scientific defaults.
 */
export interface ParsedUrlState {
  target: string | null;
  ra: number | null;
  dec: number | null;
  fov: number;
  mode: ComparisonMode;
  wlMin: number;
  wlMax: number;
}

/**
 * Patch accepted by `serializeUrlState` / `replaceUrlState`.
 *
 * - `undefined` leaves an existing query parameter untouched.
 * - `null` removes the parameter.
 * - A value writes (and normalizes) the parameter.
 */
export interface UrlStatePatch {
  target?: string | null;
  ra?: number | null;
  dec?: number | null;
  fov?: number | null;
  mode?: ComparisonMode | string | null;
  wlMin?: number | null;
  wlMax?: number | null;
}

export interface UrlHistoryLike {
  state?: unknown;
  replaceState(data: unknown, unused: string, url?: string | null): void;
}

export interface UrlLocationLike {
  pathname?: string;
  search?: string;
  hash?: string;
}

export interface ReplaceUrlStateOptions {
  baseSearch?: string;
  history?: UrlHistoryLike | null;
  location?: UrlLocationLike | null;
}

const KNOWN_PARAM_KEYS = ['target', 'ra', 'dec', 'fov', 'mode', 'wlMin', 'wlMax'] as const;

const WL_MIN_LIMIT = 0.65;
const WL_MAX_LIMIT = 5.0;
const FOV_MAX = 180.0;

function toSearchParams(search?: string | URLSearchParams | null): URLSearchParams {
  if (search instanceof URLSearchParams) {
    return new URLSearchParams(search);
  }
  let raw = (search ?? '').trim();
  if (!raw) return new URLSearchParams();

  if (raw.startsWith('?')) {
    raw = raw.slice(1);
  } else if (raw.includes('?')) {
    raw = raw.slice(raw.indexOf('?') + 1);
  }
  if (raw.includes('#')) {
    raw = raw.slice(0, raw.indexOf('#'));
  }
  return new URLSearchParams(raw);
}

function parseFiniteNumber(raw: string | null): number | null {
  if (raw === null) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

function parseNumberInRange(raw: string | null, min: number, max: number): number | null {
  const value = parseFiniteNumber(raw);
  if (value === null || value < min || value > max) return null;
  return value;
}

function isComparisonMode(value: string): value is ComparisonMode {
  return (SUPPORTED_COMPARISON_MODES as readonly string[]).includes(value);
}

function sanitizeTarget(raw: string | null): string | null {
  if (raw === null) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  // Preset slugs / catalog identifiers only; reject markup or whitespace.
  return /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(trimmed) ? trimmed : null;
}

function formatNumber(value: number, decimals: number): string {
  const rounded = Number(value.toFixed(decimals));
  return String(rounded);
}

/**
 * Parse `?target=...&ra=...&dec=...&fov=...&mode=...&wlMin=...&wlMax=...`
 * from a raw search string, a full URL, or `URLSearchParams`.
 *
 * Invalid or out-of-range values are ignored in favour of safe defaults.
 */
export function parseUrlState(search?: string | URLSearchParams | null): ParsedUrlState {
  const resolvedSearch = search ?? (typeof window === 'undefined' ? '' : window.location.search);
  const params = toSearchParams(resolvedSearch);

  const rawMode = (params.get('mode') ?? '').trim().toLowerCase();
  const mode: ComparisonMode = isComparisonMode(rawMode) ? rawMode : URL_STATE_DEFAULTS.mode;

  let ra = parseNumberInRange(params.get('ra'), 0, 360);
  if (ra === 360) ra = 0;
  const dec = parseNumberInRange(params.get('dec'), -90, 90);
  const fov =
    parseNumberInRange(params.get('fov'), Number.EPSILON, FOV_MAX) ?? URL_STATE_DEFAULTS.fov;

  let wlMin =
    parseNumberInRange(params.get('wlMin'), WL_MIN_LIMIT, WL_MAX_LIMIT) ?? URL_STATE_DEFAULTS.wlMin;
  let wlMax =
    parseNumberInRange(params.get('wlMax'), WL_MIN_LIMIT, WL_MAX_LIMIT) ?? URL_STATE_DEFAULTS.wlMax;
  if (wlMin > wlMax) {
    [wlMin, wlMax] = [wlMax, wlMin];
  }

  return {
    target: sanitizeTarget(params.get('target')),
    ra,
    dec,
    fov,
    mode,
    wlMin,
    wlMax,
  };
}

function applyTarget(params: URLSearchParams, value: UrlStatePatch['target']): void {
  if (value === undefined) return;
  if (value === null) {
    params.delete('target');
    return;
  }
  const target = sanitizeTarget(value);
  if (target) params.set('target', target);
  else params.delete('target');
}

function applyBoundedNumber(
  params: URLSearchParams,
  key: string,
  value: number | null | undefined,
  min: number,
  max: number,
  decimals: number,
): void {
  if (value === undefined) return;
  if (value === null || !Number.isFinite(value) || value < min || value > max) {
    params.delete(key);
    return;
  }
  let normalized = value;
  if (key === 'ra' && normalized === 360) normalized = 0;
  params.set(key, formatNumber(normalized, decimals));
}

/**
 * Serialize a patch into a URL query string **without** the leading `?`.
 *
 * Existing/non-sky parameters in `baseSearch` are preserved, while sky
 * parameters omitted from the patch keep their previous value. Passing `null`
 * removes a parameter.
 */
export function serializeUrlState(
  state: UrlStatePatch,
  baseSearch?: string | URLSearchParams | null,
): string {
  const merged = toSearchParams(baseSearch);

  applyTarget(merged, state.target);
  applyBoundedNumber(merged, 'ra', state.ra, 0, 360, 4);
  applyBoundedNumber(merged, 'dec', state.dec, -90, 90, 4);
  applyBoundedNumber(merged, 'fov', state.fov, Number.EPSILON, FOV_MAX, 2);

  if (state.mode !== undefined) {
    if (state.mode === null) {
      merged.delete('mode');
    } else {
      const mode = String(state.mode).trim().toLowerCase();
      if (isComparisonMode(mode)) merged.set('mode', mode);
      else merged.delete('mode');
    }
  }

  applyBoundedNumber(merged, 'wlMin', state.wlMin, WL_MIN_LIMIT, WL_MAX_LIMIT, 2);
  applyBoundedNumber(merged, 'wlMax', state.wlMax, WL_MIN_LIMIT, WL_MAX_LIMIT, 2);

  // Keep the wavelength window ordered even when only one edge was patched.
  const min = parseFiniteNumber(merged.get('wlMin'));
  const max = parseFiniteNumber(merged.get('wlMax'));
  if (min !== null && max !== null && min > max) {
    merged.set('wlMin', formatNumber(max, 2));
    merged.set('wlMax', formatNumber(min, 2));
  }

  // Canonical ordering: known sky parameters first, unrelated params after.
  const ordered = new URLSearchParams();
  for (const key of KNOWN_PARAM_KEYS) {
    const value = merged.get(key);
    if (value !== null) ordered.set(key, value);
  }
  for (const [key, value] of merged) {
    if (!(KNOWN_PARAM_KEYS as readonly string[]).includes(key)) {
      ordered.append(key, value);
    }
  }

  return ordered.toString();
}

/**
 * Update the browser query string in-place via `history.replaceState` (never a
 * page navigation). Returns the relative URL that was written.
 */
export function replaceUrlState(
  state: UrlStatePatch,
  options: ReplaceUrlStateOptions = {},
): string {
  const browserWindow = typeof window === 'undefined' ? null : window;
  const location: UrlLocationLike | null =
    options.location ?? (browserWindow ? browserWindow.location : null);
  const history: UrlHistoryLike | null =
    options.history ?? (browserWindow ? browserWindow.history : null);

  const baseSearch = options.baseSearch ?? location?.search ?? '';
  const query = serializeUrlState(state, baseSearch);
  const pathname = location?.pathname ?? '/';
  const hash = location?.hash ?? '';
  const url = `${pathname}${query ? `?${query}` : ''}${hash}`;
  const currentUrl = `${pathname}${location?.search ?? ''}${hash}`;

  if (history && url !== currentUrl) {
    history.replaceState(history.state ?? null, '', url);
  }

  return url;
}
