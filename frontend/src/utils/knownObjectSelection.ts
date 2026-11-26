/**
 * Pure helpers for the Viewport Controls Bar and Known Celestial Objects menu (ISS-16).
 *
 * Kept free of React so the selection, formatting and wavelength-preset logic can be unit
 * tested with `node --test` and reused identically by the UI and SkyExplorerContext.
 */

import {
  TRACKING_SQUARE_COLORS,
  type CatalogSourceType,
  type CelestialObjectClass,
  type GlobalKnownObjectEntry,
  type KnownCelestialObject,
  type MissionName,
} from '../types/contracts.ts';

/** Preset wavelength slice option shown in the Blinker sidebar quick-filter pills. */
export interface WavelengthPreset {
  id: string;
  label: string;
  shortLabel: string;
  range: [number, number];
}

/** Standard SPHEREx infrared Dim & Pop quick-filter wavelength presets. */
export const WAVELENGTH_PRESETS: ReadonlyArray<WavelengthPreset> = [
  { id: 'all', label: 'All Bands (0.75–5.0 µm)', shortLabel: 'All 0.75-5.0', range: [0.75, 5.0] },
  {
    id: 'cold',
    label: 'Cold Objects / Planet X (3.8–5.0 µm)',
    shortLabel: 'Cold 3.8-5.0',
    range: [3.8, 5.0],
  },
  {
    id: 'molecules',
    label: 'H₂O / CH₄ Band (2.4–3.8 µm)',
    shortLabel: 'Ice 2.4-3.8',
    range: [2.4, 3.8],
  },
  { id: 'hot', label: 'Hot Stars (0.75–2.4 µm)', shortLabel: 'Hot 0.75-2.4', range: [0.75, 2.4] },
];

/** Mission-specific wavelength slider bounds, step, and quick-filter presets. */
export interface MissionWavelengthConfig {
  isMissionRestricted: boolean;
  /** Full spectrum rail lower bound (`0.65 µm`). */
  railMin: number;
  /** Full spectrum rail upper bound (`5.0 µm`). */
  railMax: number;
  /** Mission-clamped minimum thumb bound in µm. */
  min: number;
  /** Mission-clamped maximum thumb bound in µm. */
  max: number;
  step: number;
  minGap: number;
  defaultRange: [number, number];
  presets: ReadonlyArray<WavelengthPreset>;
}

const RAIL_MIN_UM = 0.65;
const RAIL_MAX_UM = 5.0;

const SPHEREX_WAVELENGTH_CONFIG: MissionWavelengthConfig = {
  isMissionRestricted: false,
  railMin: RAIL_MIN_UM,
  railMax: RAIL_MAX_UM,
  min: 0.75,
  max: 5.0,
  step: 0.05,
  minGap: 0.15,
  defaultRange: [0.75, 5.0],
  presets: WAVELENGTH_PRESETS,
};

const SKYVIEW_WAVELENGTH_CONFIG: MissionWavelengthConfig = {
  isMissionRestricted: true,
  railMin: RAIL_MIN_UM,
  railMax: RAIL_MAX_UM,
  min: 0.65,
  max: 0.85,
  step: 0.05,
  minGap: 0.05,
  defaultRange: [0.65, 0.85],
  presets: WAVELENGTH_PRESETS,
};

const WISE_WAVELENGTH_CONFIG: MissionWavelengthConfig = {
  isMissionRestricted: true,
  railMin: RAIL_MIN_UM,
  railMax: RAIL_MAX_UM,
  min: 2.4,
  max: 5.0,
  step: 0.05,
  minGap: 0.15,
  defaultRange: [2.4, 5.0],
  presets: WAVELENGTH_PRESETS,
};

/**
 * Resolve the wavelength slider bounds for the active mission(s) on the full
 * `0.65–5.0 µm` spectrum rail.
 *
 * - `SkyView`-only: knobs clamp to `[0.65, 0.85]` µm (DSS2 Red/IR).
 * - `WISE`-only: knobs clamp to `[2.4, 5.0]` µm (W1/W2).
 * - `SPHEREx` / mixed / default: knobs clamp to `[0.75, 5.0]` µm.
 *
 * @param availableMissions - Missions present in the loaded patch manifest.
 * @param activeMissions - Currently enabled mission filter pills.
 * @param primaryMission - Optional `data_source_summary.primary_mission` fallback.
 * @returns Resolved {@link MissionWavelengthConfig}.
 */
export function resolveWavelengthConfigForMissions(
  availableMissions: ReadonlyArray<MissionSourceCount>,
  activeMissions: ReadonlyArray<MissionName>,
  primaryMission?: MissionName,
): MissionWavelengthConfig {
  const enabledInPatch = availableMissions
    .filter((entry) => activeMissions.includes(entry.mission))
    .map((entry) => entry.mission);

  const soleMission =
    enabledInPatch.length === 1
      ? enabledInPatch[0]
      : enabledInPatch.length === 0
        ? primaryMission
        : undefined;

  if (soleMission === 'SkyView') return SKYVIEW_WAVELENGTH_CONFIG;
  if (soleMission === 'WISE') return WISE_WAVELENGTH_CONFIG;
  return SPHEREX_WAVELENGTH_CONFIG;
}

/**
 * Determine whether a wavelength quick-filter preset falls within the active
 * mission's allowed `[min, max]` bounds.
 *
 * @param range - Preset `[minUm, maxUm]` bounds.
 * @param config - Active {@link MissionWavelengthConfig}.
 * @returns `true` if the preset is selectable for the active mission.
 */
export function isWavelengthPresetEnabled(
  range: readonly [number, number],
  config: Pick<MissionWavelengthConfig, 'min' | 'max'>,
): boolean {
  return range[0] >= config.min - 1e-6 && range[1] <= config.max + 1e-6;
}

/**
 * Clamp or reset a wavelength `[minUm, maxUm]` range when switching between
 * mission wavelength regimes (e.g. SPHEREx <-> SkyView <-> WISE).
 *
 * @param currentRange - Previously active `[minUm, maxUm]` range.
 * @param nextConfig - Target mission's {@link MissionWavelengthConfig}.
 * @param prevConfig - Previous mission's {@link MissionWavelengthConfig}, if known.
 * @returns Validated `[minUm, maxUm]` range within `nextConfig`.
 */
export function reconcileWavelengthRange(
  currentRange: readonly [number, number],
  nextConfig: MissionWavelengthConfig,
  prevConfig?: MissionWavelengthConfig,
): [number, number] {
  if (prevConfig && (prevConfig.min !== nextConfig.min || prevConfig.max !== nextConfig.max)) {
    return [...nextConfig.defaultRange];
  }
  const [lo, hi] = currentRange;
  if (
    !Number.isFinite(lo) ||
    !Number.isFinite(hi) ||
    lo < nextConfig.min - 1e-6 ||
    hi > nextConfig.max + 1e-6 ||
    (nextConfig.max > 0.85 + 1e-6 && hi <= 0.85 + 1e-6)
  ) {
    return [...nextConfig.defaultRange];
  }
  return [lo, hi];
}

const PRESET_MATCH_TOLERANCE_UM = 0.05;

/**
 * Determine whether the active wavelength slider range matches a given preset pill.
 *
 * @param current - Active `[minUm, maxUm]` wavelength slice.
 * @param range - Preset `[minUm, maxUm]` bounds.
 * @returns `true` when both bounds match within `0.05 µm`.
 */
export function isWavelengthPresetActive(
  current: readonly [number, number],
  range: readonly [number, number],
): boolean {
  return (
    Math.abs(current[0] - range[0]) < PRESET_MATCH_TOLERANCE_UM &&
    Math.abs(current[1] - range[1]) < PRESET_MATCH_TOLERANCE_UM
  );
}

const CATALOG_BADGE_LABELS: Record<CatalogSourceType, string> = {
  gaia_dr3: 'Gaia DR3',
  wise: 'WISE',
  most: 'MOST',
  simbad: 'SIMBAD',
  jpl_horizons: 'JPL Horizons',
};

/**
 * Format a catalog source identifier into a human-readable badge label.
 *
 * @param catalog - Canonical catalog source identifier.
 * @returns Display badge string (e.g., `'Gaia DR3'`).
 */
export function catalogBadgeLabel(catalog: CatalogSourceType): string {
  return CATALOG_BADGE_LABELS[catalog] ?? catalog;
}

const OBJECT_CLASS_LABELS: Record<CelestialObjectClass, string> = {
  star: 'Star',
  brown_dwarf: 'Brown Dwarf',
  planet: 'Planet',
  moon: 'Moon',
  comet: 'Comet',
  asteroid: 'Asteroid',
  galaxy: 'Galaxy',
  nebula: 'Nebula',
  variable_star: 'Variable Star',
  infrared_source: 'IR Source',
};

/**
 * Infer a normalized {@link CelestialObjectClass} from object metadata and label hints.
 *
 * @param obj - Partial object record containing class, type, and label.
 * @returns Resolved celestial object classification.
 */
export function inferObjectClass(
  obj: Pick<KnownCelestialObject, 'object_class' | 'object_type' | 'label'>,
): CelestialObjectClass {
  if (obj.object_class) {
    return obj.object_class;
  }
  const low = obj.label.toLowerCase();
  if (obj.object_type === 'brown_dwarf_candidate') return 'brown_dwarf';
  if (obj.object_type === 'solar_system') {
    if (low.includes('comet') || low.includes('3i') || low.includes('atlas')) return 'comet';
    if (low.includes('ceres') || low.includes('vesta') || low.includes('apophis'))
      return 'asteroid';
    if (
      low.includes('io') ||
      low.includes('europa') ||
      low.includes('ganymede') ||
      low.includes('callisto') ||
      low.includes('titan')
    ) {
      return 'moon';
    }
    return 'planet';
  }
  if (obj.object_type === 'variable_star') return 'variable_star';
  if (obj.object_type === 'infrared_source') return 'infrared_source';
  return 'star';
}

/**
 * Format a {@link CelestialObjectClass} enum value into a display badge label.
 *
 * @param objClass - Celestial object classification.
 * @returns Human-readable class label.
 */
export function objectClassBadgeLabel(objClass: CelestialObjectClass): string {
  return OBJECT_CLASS_LABELS[objClass] ?? objClass;
}

/**
 * Determine whether a known object is a moving Solar System body or exhibits
 * measurable multi-epoch pixel motion (`>= 0.5 px`).
 *
 * @param obj - Known celestial object entry.
 * @returns `true` if the object is classified or observed as moving across epochs.
 */
export function isMovingSolarSystemObject(obj: KnownCelestialObject): boolean {
  const cls = inferObjectClass(obj);
  if (cls === 'planet' || cls === 'moon' || cls === 'comet' || cls === 'asteroid') {
    return true;
  }
  if (obj.object_type === 'solar_system') {
    return true;
  }
  if (obj.motion_vector) {
    const pxDisp = Math.hypot(obj.motion_vector.dx_px, obj.motion_vector.dy_px);
    if (pxDisp >= 0.5) return true;
  }
  if (obj.epoch_positions.length >= 2) {
    const first = obj.epoch_positions[0];
    const last = obj.epoch_positions[obj.epoch_positions.length - 1];
    if (Math.hypot(last.x - first.x, last.y - first.y) >= 0.5) {
      return true;
    }
  }
  return false;
}

/**
 * Format an apparent magnitude into a compact UI label.
 *
 * @param magnitude - Apparent magnitude, or `null` when unavailable.
 * @returns Formatted magnitude string (`'mag 12.3'` or `'mag n/a'`).
 */
export function formatMagnitude(magnitude: number | null): string {
  return magnitude === null || !Number.isFinite(magnitude)
    ? 'mag n/a'
    : `mag ${magnitude.toFixed(1)}`;
}

/**
 * Format total proper motion in arcseconds per year; sub-arcsecond values retain
 * 3 decimals so Gaia field stars remain readable.
 *
 * @param arcsecPerYr - Total proper motion in arcsec/yr, or `null` when unavailable.
 * @returns Formatted proper-motion label.
 */
export function formatProperMotion(arcsecPerYr: number | null): string {
  if (arcsecPerYr === null || !Number.isFinite(arcsecPerYr)) return 'μ n/a';
  return `μ ${arcsecPerYr.toFixed(arcsecPerYr < 1 ? 3 : 2)}″/yr`;
}

/**
 * Compute default checked object IDs from `knownObjects`, keeping only
 * items marked `default_checked = true`.
 *
 * @param knownObjects - Known celestial objects in the active sky patch.
 * @returns Array of default-checked object IDs.
 */
export function getDefaultCheckedObjectIds(
  knownObjects: ReadonlyArray<Pick<KnownCelestialObject, 'object_id' | 'default_checked'>>,
): string[] {
  return knownObjects.filter((obj) => obj.default_checked).map((obj) => obj.object_id);
}

/**
 * Randomly select `count` distinct highlighter colors from `TRACKING_SQUARE_COLORS`,
 * prioritizing colors not currently in `usedColors` before cycling.
 *
 * @param count - Number of colors to pick.
 * @param usedColors - Colors already reserved or in use.
 * @param rng - Optional uniform `[0, 1)` random number generator for testing.
 * @returns Array of `count` hex color strings.
 */
export function pickDistinctHighlighterColors(
  count: number,
  usedColors: Iterable<string> = [],
  rng: () => number = Math.random,
): string[] {
  const usedSet = new Set(usedColors);
  const available = TRACKING_SQUARE_COLORS.filter((c) => !usedSet.has(c));
  const shuffle = (arr: string[]): string[] => {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = copy[i];
      copy[i] = copy[j];
      copy[j] = tmp;
    }
    return copy;
  };

  const result: string[] = [];
  let pool = shuffle(available);
  while (result.length < count) {
    if (pool.length === 0) {
      pool = shuffle([...TRACKING_SQUARE_COLORS]);
    }
    const nextColor = pool.pop();
    if (nextColor) {
      result.push(nextColor);
    }
  }
  return result;
}

/**
 * Return all `object_id` strings from a known-object list for "Check All".
 *
 * @param knownObjects - Known objects in the active patch.
 * @returns Array of all object IDs.
 */
export function selectAllObjectIds(knownObjects: ReadonlyArray<{ object_id: string }>): string[] {
  return knownObjects.map((obj) => obj.object_id);
}

/**
 * Return an empty checked-object ID list for "Uncheck All".
 *
 * @returns Empty string array.
 */
export function clearCheckedObjectIds(): string[] {
  return [];
}

/**
 * Immutably toggle membership of `objectId` in `checkedIds`.
 *
 * @param checkedIds - Currently checked object IDs.
 * @param objectId - Object ID to toggle.
 * @returns New array with `objectId` added or removed.
 */
export function toggleObjectId(checkedIds: readonly string[], objectId: string): string[] {
  return checkedIds.includes(objectId)
    ? checkedIds.filter((id) => id !== objectId)
    : [...checkedIds, objectId];
}

/** Per-mission epoch count summary extracted from a patch manifest. */
export interface MissionSourceCount {
  mission: MissionName;
  count: number;
}

/**
 * Summarize distinct missions present in `epochs` alongside their frame counts.
 *
 * @param epochs - Epoch entries from a patch manifest.
 * @returns Ordered list of `{ mission, count }` summaries.
 */
export function extractAvailableMissions(
  epochs: ReadonlyArray<{ mission: MissionName }>,
): MissionSourceCount[] {
  const counts = new Map<MissionName, number>();
  for (const ep of epochs) {
    counts.set(ep.mission, (counts.get(ep.mission) ?? 0) + 1);
  }
  return Array.from(counts.entries()).map(([mission, count]) => ({ mission, count }));
}

/**
 * Resolve epoch indices belonging to `activeMissions`, falling back to all
 * epochs if the filter would otherwise hide every frame.
 *
 * @param epochs - Epoch entries from a patch manifest.
 * @param activeMissions - Currently enabled mission names.
 * @returns Non-empty list of permitted epoch indices.
 */
export function resolveFilteredEpochIndices(
  epochs: ReadonlyArray<{ mission: MissionName }>,
  activeMissions: ReadonlyArray<MissionName>,
): number[] {
  if (!epochs || epochs.length === 0) return [];
  const indices: number[] = [];
  epochs.forEach((ep, idx) => {
    if (activeMissions.includes(ep.mission)) {
      indices.push(idx);
    }
  });
  return indices.length > 0 ? indices : epochs.map((_, idx) => idx);
}

/**
 * Pick `currentIndex` if allowed, otherwise the closest index in `allowedIndices`.
 *
 * @param currentIndex - Current active epoch index.
 * @param allowedIndices - Permitted epoch indices.
 * @returns Nearest allowed epoch index (or `0` when `allowedIndices` is empty).
 */
export function pickNearestAllowedEpoch(
  currentIndex: number,
  allowedIndices: readonly number[],
): number {
  if (allowedIndices.length === 0) return 0;
  if (allowedIndices.includes(currentIndex)) return currentIndex;
  let best = allowedIndices[0];
  let bestDist = Math.abs(best - currentIndex);
  for (const idx of allowedIndices) {
    const dist = Math.abs(idx - currentIndex);
    if (dist < bestDist) {
      bestDist = dist;
      best = idx;
    }
  }
  return best;
}

/**
 * Pick the epoch index whose Modified Julian Date is closest to `targetMjd`.
 *
 * @param epochs - Ordered manifest epochs containing `mjd`.
 * @param targetMjd - Target Modified Julian Date.
 * @returns Zero-based index of the closest epoch, or `0` when empty/non-finite.
 */
export function pickClosestEpochIndex(
  epochs: ReadonlyArray<{ mjd: number }>,
  targetMjd: number,
): number {
  if (!epochs || epochs.length === 0 || !Number.isFinite(targetMjd)) return 0;
  let bestIdx = 0;
  let bestDist = Math.abs(epochs[0].mjd - targetMjd);
  for (let i = 1; i < epochs.length; i += 1) {
    const dist = Math.abs(epochs[i].mjd - targetMjd);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Compute a normalized `[0.0, 1.0]` fuzzy match score between a user query and a candidate string.
 * Supports exact/prefix/substring match, multi-word token prefix match, ordered subsequence
 * (`"w0855"` -> `"WISE 0855-0714"`), and bounded Levenshtein edit distance (`"barnrd"` -> `"Barnard's Star"`).
 *
 * @param query - User search string.
 * @param candidate - Candidate object name or identifier.
 * @returns Match score in `[0.0, 1.0]`.
 */
export function fuzzyScoreText(query: string, candidate: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 1.0;
  const c = candidate.trim().toLowerCase();
  if (!c) return 0.0;
  if (q === c) return 1.0;
  if (c.startsWith(q)) return 0.96;
  const subIdx = c.indexOf(q);
  if (subIdx >= 0) {
    return 0.9 - Math.min(0.12, (subIdx / Math.max(1, c.length)) * 0.12);
  }

  const qTokens = q.replace(/[-_']/g, ' ').split(/\s+/).filter(Boolean);
  const cTokens = c.replace(/[-_']/g, ' ').split(/\s+/).filter(Boolean);
  if (
    qTokens.length > 0 &&
    qTokens.every((qt) => cTokens.some((ct) => ct.startsWith(qt) || ct.includes(qt)))
  ) {
    return 0.84;
  }

  // Ordered subsequence match
  let qIdx = 0;
  let gaps = 0;
  let lastMatch = -1;
  for (let i = 0; i < c.length && qIdx < q.length; i += 1) {
    if (c[i] === q[qIdx]) {
      if (lastMatch >= 0) {
        gaps += i - lastMatch - 1;
      }
      lastMatch = i;
      qIdx += 1;
    }
  }
  if (qIdx === q.length && q.length >= 3) {
    return Math.max(0.52, 0.78 - Math.min(0.24, gaps * 0.02));
  }

  // Bounded Levenshtein edit distance against candidate tokens for typo tolerance
  let bestTokenScore = 0.0;
  for (const ct of cTokens) {
    if (Math.abs(ct.length - q.length) > 3) continue;
    let prev = Array.from({ length: ct.length + 1 }, (_, idx) => idx);
    for (let i = 1; i <= q.length; i += 1) {
      const curr = [i];
      for (let j = 1; j <= ct.length; j += 1) {
        const cost = q[i - 1] === ct[j - 1] ? 0 : 1;
        curr.push(Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost));
      }
      prev = curr;
    }
    const dist = prev[ct.length];
    const maxLen = Math.max(q.length, ct.length, 1);
    const sim = 1.0 - dist / maxLen;
    if (sim >= 0.65) {
      bestTokenScore = Math.max(bestTokenScore, sim * 0.74);
    }
  }
  return bestTokenScore;
}

/**
 * Filter and rank {@link KnownCelestialObject} / {@link GlobalKnownObjectEntry} items
 * using {@link fuzzyScoreText}.
 *
 * @param objects - Candidate objects to search.
 * @param query - User search string (empty returns all objects in input order).
 * @returns Ranked matching objects sorted by relevance, default-checked status, and brightness.
 */
export function fuzzySearchKnownObjects<T extends KnownCelestialObject | GlobalKnownObjectEntry>(
  objects: readonly T[],
  query: string,
): T[] {
  const clean = query.trim();
  if (!clean) {
    return [...objects];
  }
  const scored: { item: T; score: number }[] = [];
  for (const obj of objects) {
    const entry = obj as GlobalKnownObjectEntry;
    const fields = [
      obj.label,
      obj.canonical_name ?? '',
      obj.otype ?? '',
      obj.object_class ?? '',
      obj.object_id,
      obj.catalog,
      obj.object_type.replace(/_/g, ' '),
      entry.target_name ?? '',
      entry.target_slug ?? '',
      entry.constellation ?? '',
    ];
    let best = 0;
    for (const f of fields) {
      if (!f) continue;
      const s = fuzzyScoreText(clean, f);
      if (s > best) best = s;
    }
    if (best >= 0.45) {
      scored.push({ item: obj, score: best });
    }
  }
  scored.sort((a, b) => {
    if (Math.abs(b.score - a.score) > 1e-4) return b.score - a.score;
    const aPrim = a.item.default_checked ? 1 : 0;
    const bPrim = b.item.default_checked ? 1 : 0;
    if (bPrim !== aPrim) return bPrim - aPrim;
    return (a.item.magnitude ?? 20) - (b.item.magnitude ?? 20);
  });
  return scored.map((s) => s.item);
}

/**
 * Compute a deterministic 32-bit FNV-1a hash for an object identifier.
 *
 * @param value - Object identifier string.
 * @returns Unsigned 32-bit integer hash.
 */
function hashObjectId(value: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/**
 * Deterministically assign distinct highlighter palette colors to known objects,
 * preserving any already-assigned `object_id -> color` entries from `existingColors`
 * across progressive Stage A catalog -> Stage B cutout updates.
 *
 * @param knownObjects - Known celestial objects in the active sky patch.
 * @param existingColors - Previously assigned `object_id -> color` map to preserve.
 * @returns Deterministic `object_id -> hexColor` map with zero randomness.
 */
export function assignDeterministicHighlighterColors(
  knownObjects: ReadonlyArray<Pick<KnownCelestialObject, 'object_id'>>,
  existingColors: Readonly<Record<string, string>> = {},
): Record<string, string> {
  if (knownObjects.length === 0) return {};

  const palette = TRACKING_SQUARE_COLORS;
  const paletteSize = palette.length;
  const result: Record<string, string> = {};
  const usedColors = new Set<string>();

  for (const obj of knownObjects) {
    const preserved = existingColors[obj.object_id];
    if (preserved) {
      result[obj.object_id] = preserved;
      usedColors.add(preserved);
    }
  }

  for (const obj of knownObjects) {
    if (result[obj.object_id]) continue;
    const baseIdx = hashObjectId(obj.object_id) % paletteSize;
    let chosen: string | null = null;
    for (let offset = 0; offset < paletteSize; offset += 1) {
      const candidate = palette[(baseIdx + offset) % paletteSize];
      if (!usedColors.has(candidate)) {
        chosen = candidate;
        break;
      }
    }
    const finalColor = chosen ?? palette[baseIdx];
    result[obj.object_id] = finalColor;
    usedColors.add(finalColor);
  }

  return result;
}
