/**
 * Shared TypeScript domain contracts and binary/spectral invariants for
 * Star Hunters (Planet X & SPHEREx Blinker).
 *
 * Kept strictly pure (types, interfaces, and constants only; zero runtime
 * helpers or re-exports) and executable under Node 24's type-stripping mode.
 */

import worldCapitalCities from './world_capital_cities.json' with { type: 'json' };

/** Supported infrared and optical sky-survey missions. */
export type MissionName =
  'SPHEREx' | 'WISE' | 'SkyView' | '2MASS' | 'DSS2' | 'Horizons' | 'NEOSSat' | 'Simulated';

/** Wire encoding precision for binary 6-detector cutout frames. */
export type BinaryDtype = 'float32' | 'float16';

/** User-configurable fetch quality mode for sky patch binary downloads. */
export type FetchQualityMode = 'auto' | 'full' | 'compressed';

/** Metadata persisted for a recently opened Blinker sky patch. */
export interface RecentPatchEntry {
  id: string;
  kind: 'preset' | 'live';
  slug?: string | null;
  ra: number;
  dec: number;
  fovArcmin: number;
  title: string;
  coordsLabel: string;
  mission?: string;
  openedAt: number;
}

/** Multi-epoch pixel comparison mode in the Blinker viewport. */
export type ComparisonMode = 'blink' | 'subtract' | 'coadd';

/** Tone-mapping curve applied to weighted flux values. */
export type ContrastStretch = 'asinh' | 'linear' | 'zscale';

/** Astrometric / photometric catalog provenance identifiers. */
export type CatalogSourceType = 'gaia_dr3' | 'wise' | 'most' | 'simbad' | 'jpl_horizons';

/** Coarse catalog object category emitted by backend resolvers. */
export type CelestialObjectCategory =
  | 'star'
  | 'high_pm_star'
  | 'brown_dwarf_candidate'
  | 'solar_system'
  | 'variable_star'
  | 'infrared_source';

/** Fine-grained astrophysical classification used for UI badges and markers. */
export type CelestialObjectClass =
  | 'star'
  | 'brown_dwarf'
  | 'planet'
  | 'moon'
  | 'comet'
  | 'asteroid'
  | 'galaxy'
  | 'nebula'
  | 'variable_star'
  | 'infrared_source';

/** Width of each cutout detector plane in pixels. */
export const GRID_WIDTH = 256;

/** Height of each cutout detector plane in pixels. */
export const GRID_HEIGHT = 256;

/** Number of SPHEREx spectral detector bands per epoch cube. */
export const NUM_DETECTORS = 6;

/** Total pixels in a single 256x256 detector plane (`65,536`). */
export const PIXELS_PER_PLANE = GRID_WIDTH * GRID_HEIGHT;

/** Total scalar values in a 6-detector epoch cube (`393,216`). */
export const TOTAL_DETECTOR_VALUES = NUM_DETECTORS * PIXELS_PER_PLANE;

/** Expected byte length of an uncompressed `float32` detector cube (`1,572,864`). */
export const BINARY_DETECTOR_FLOAT32_BYTES = TOTAL_DETECTOR_VALUES * 4;

/** Expected byte length of a 3G-compressed `float16` detector cube (`786,432`). */
export const BINARY_DETECTOR_FLOAT16_BYTES = TOTAL_DETECTOR_VALUES * 2;

/** Expected byte length of an 8-bit pixel hygiene mask (`65,536`). */
export const BINARY_FLAGS_UINT8_BYTES = PIXELS_PER_PLANE;

/** Inclusive wavelength bounds (micrometres) for SPHEREx Detectors 1–6. */
export const DETECTOR_WAVELENGTH_BOUNDS_UM: ReadonlyArray<readonly [number, number]> = [
  [0.75, 1.1], // Detector 1
  [1.1, 1.6], // Detector 2
  [1.6, 2.42], // Detector 3
  [2.42, 3.8], // Detector 4 (WISE W1 3.4 um fallback)
  [3.8, 4.4], // Detector 5 (WISE W2 4.6 um fallback)
  [4.4, 5.0], // Detector 6 (WISE W2 4.6 um fallback)
];

/**
 * Distinct high-contrast 20-color palette used for object highlighters and
 * tracking squares, aligned verbatim with `backend/app/models/contracts.py`.
 */
export const TRACKING_SQUARE_COLORS: readonly string[] = [
  '#00F5D4', // Cyan-Mint
  '#FFBE0B', // Solar Amber
  '#FF006E', // Magenta Pulse
  '#3A86FF', // Azure Blue
  '#FB5607', // Neon Tangerine
  '#8338EC', // Ultraviolet
  '#06D6A0', // Emerald
  '#FF9F1C', // Gold
  '#AACC00', // Electric Lime
  '#FF4D6D', // Coral Red
  '#F72585', // Hot Pink
  '#4CC9F0', // Sky Cyan
  '#80ED99', // Mint Green
  '#C77DFF', // Orchid Violet
  '#FFEA00', // Laser Yellow
  '#00BBF9', // Cerulean
  '#FF6B6B', // Salmon Rose
  '#72EFDD', // Aquamarine
  '#F15BB5', // Bubblegum Magenta
  '#FEE440', // Starlight Yellow
] as const;

/**
 * Default molecular absorption bands (micrometres) overlaid on the 102-band
 * SPHEREx spectrum plot when a payload omits `molecular_features`.
 */
export const DEFAULT_MOLECULAR_BANDS: ReadonlyArray<MolecularFeature> = [
  { molecule: 'h2o', min_um: 2.95, max_um: 3.3, label: 'H2O Ice/Vapor' },
  { molecule: 'ch4', min_um: 3.2, max_um: 3.55, label: 'CH4 Methane' },
  { molecule: 'co2', min_um: 4.2, max_um: 4.35, label: 'CO2' },
  { molecule: 'co', min_um: 4.55, max_um: 4.8, label: 'CO' },
];

/** World capital city entry used for offline/online Earth-to-Sky zenith lookup. */
export interface WorldCapitalCity {
  slug: string;
  city_name: string;
  country: string;
  lat: number;
  lon: number;
}

/** Static catalog of world capital coordinates bundled with the client. */
export const WORLD_CAPITAL_CITIES: ReadonlyArray<WorldCapitalCity> = worldCapitalCities;

/** Per-epoch detection outcome for a click-tracked target waypoint. */
export type WaypointTrackingStatus = 'tracked' | 'lost' | 'out_of_bounds' | 'low_snr';

/** Overall multi-epoch tracking summary status. */
export type TrackSummaryStatus = 'tracked' | 'partially_lost' | 'no_body_found';

/** Quantized sky-patch query metadata returned by backend endpoints. */
export interface QueryInfo {
  ra: number;
  dec: number;
  fov_arcmin: number;
  grid_key: string;
  cached: boolean;
}

/** Human-readable target metadata for a sky patch or preset. */
export interface TargetMetadata {
  slug: string;
  name: string;
  category: string;
  description: string;
  constellation: string;
}

/** Mission provenance and temporal baseline summary for a sky patch. */
export interface DataSourceSummary {
  primary_mission: MissionName;
  has_multi_epoch_spherex: boolean;
  uses_wise_fallback: boolean;
  baseline_days: number;
  status_note: string;
  provenance_tier?: 'primary' | 'extension' | 'simulated';
  survey_name?: string;
}

/** Formatted provenance badge metadata displayed in the Blinker and Spectra headers. */
export interface ImageProvenanceBadge {
  tier: 'primary' | 'extension' | 'simulated';
  label: string;
  shortLabel: string;
}

/** Simplified TAN-projection World Coordinate System header for a 256x256 cutout. */
export interface WcsHeader {
  width: number;
  height: number;
  crval: [number, number];
  crpix: [number, number];
  cdelt: [number, number];
}

/** Manifest entry describing a single observation epoch and its binary URLs. */
export interface EpochManifestEntry {
  epoch_id: string;
  label: string;
  mission: MissionName;
  mjd: number;
  iso_date: string;
  detectors_data_url: string;
  flags_mask_url: string;
  available_detectors: number[];
  binary_dtype?: BinaryDtype;
}

/** Propagated sky and pixel position of a known object at a specific epoch. */
export interface KnownObjectEpochPosition {
  epoch_id: string;
  mjd: number;
  ra: number;
  dec: number;
  x: number;
  y: number;
}

/** Net pixel and sky displacement vector across the epoch baseline. */
export interface MotionVector {
  dx_px: number;
  dy_px: number;
  dra_arcsec: number;
  ddec_arcsec: number;
}

/** Cataloged celestial object cross-matched inside a sky patch. */
export interface KnownCelestialObject {
  object_id: string;
  label: string;
  catalog: CatalogSourceType;
  object_type: CelestialObjectCategory;
  object_class?: CelestialObjectClass | null;
  canonical_name?: string | null;
  otype?: string | null;
  ra: number;
  dec: number;
  x: number;
  y: number;
  magnitude: number | null;
  proper_motion_arcsec_yr: number | null;
  pmra_mas_yr?: number | null;
  pmdec_mas_yr?: number | null;
  parallax_mas?: number | null;
  w1_w2_color: number | null;
  default_checked: boolean;
  epoch_positions: KnownObjectEpochPosition[];
  motion_vector?: MotionVector | null;
  data_source?: string | null;
}

/** All-sky or search-result known-object entry with optional preset provenance. */
export interface GlobalKnownObjectEntry extends KnownCelestialObject {
  target_slug?: string;
  target_name?: string;
  constellation?: string;
  patch_ra?: number;
  patch_dec?: number;
  match_score?: number;
  ephemeris_iso_date?: string | null;
}

/** Constrained epoch option (Current Time or SPHEREx pass date) for a searchable object. */
export interface ObjectObservationEpoch {
  epoch_key: string;
  kind: 'current' | 'spherex_pass';
  label: string;
  iso_date: string;
  mjd: number;
  ra: number;
  dec: number;
  constellation: string;
  source: string;
  exposure_count?: number | null;
  can_view_in_blinker: boolean;
  note: string;
}

/** Response returned by `/api/v1/catalogs/object-epochs`. */
export interface ObjectObservationEpochsResponse {
  object_id: string;
  label: string;
  is_solar_system: boolean;
  passes_status: string;
  epochs: ObjectObservationEpoch[];
}

/** Gaia DR3 astrometric overlay source record. */
export interface GaiaDr3OverlayItem {
  source_id: string;
  ra: number;
  dec: number;
  x: number;
  y: number;
  g_mag: number;
  bp_rp: number | null;
  pmra: number | null;
  pmdec: number | null;
}

/** WISE/AllWISE mid-infrared overlay source record. */
export interface WiseOverlayItem {
  source_id: string;
  ra: number;
  dec: number;
  x: number;
  y: number;
  w1_mag: number;
  w2_mag: number;
  w1_w2_color: number;
}

/** Moving Object Search Tool (MOST) / Solar System ephemeris overlay item. */
export interface MostOverlayItem {
  object_name: string;
  epoch_id: string;
  ra: number;
  dec: number;
  x: number;
  y: number;
  v_mag: number;
}

/** Combined Stage A/B catalog overlay payload for a sky patch. */
export interface CatalogOverlays {
  known_objects?: KnownCelestialObject[];
  gaia_dr3: GaiaDr3OverlayItem[];
  wise_sources: WiseOverlayItem[];
  most_solar_system: MostOverlayItem[];
  source_status?: Record<string, string> | null;
}

/** Complete Stage B sky-patch manifest returned by `/api/v1/sky-patch` or static presets. */
export interface SkyPatchManifest {
  query: QueryInfo;
  target_metadata: TargetMetadata;
  data_source_summary: DataSourceSummary;
  wcs: WcsHeader;
  epochs: EpochManifestEntry[];
  default_spectrum_url: string | null;
  catalog_overlays: CatalogOverlays;
}

/** Stage A fast catalog response returned by `/api/v1/catalogs`. */
export interface CatalogsResponse {
  query: QueryInfo;
  wcs: WcsHeader;
  catalog_overlays: CatalogOverlays;
}

/** Single channel measurement in a 102-band SPHEREx spectrum. */
export interface SpectrumChannel {
  band_index: number;
  detector: number;
  wavelength_um: number;
  flux: number;
  variance: number;
  flagged: boolean;
}

/** Molecular absorption band window annotated on the SED plot. */
export interface MolecularFeature {
  molecule: string;
  min_um: number;
  max_um: number;
  label: string;
}

/** Full 102-band spectral energy distribution payload from `/api/v1/spectrum`. */
export interface SpectrumPayload {
  ra: number;
  dec: number;
  source_label: string;
  spectrum_origin: string;
  units: {
    wavelength: 'um';
    flux: 'MJy/sr';
  };
  channels: SpectrumChannel[];
  molecular_features: MolecularFeature[];
}

/** Overhead ICRS zenith coordinate response from `/api/v1/earth-to-sky`. */
export interface EarthToSkyResponse {
  earth: {
    lat: number;
    lon: number;
    iso_time: string;
    city_name?: string | null;
    country?: string | null;
    is_fallback_capital?: boolean;
    cached?: boolean;
  };
  zenith_icrs: {
    ra: number;
    dec: number;
    constellation: string;
  };
}

/** Client-side rendering, wavelength slice, overlay, and pan/zoom state. */
export interface ViewportRenderSettings {
  mode: ComparisonMode;
  activeEpochIndex: number;
  passAIndex: number;
  passBIndex: number;
  coaddIndices: number[];
  wavelengthRangeUm: [number, number];
  contrastStretch: ContrastStretch;
  showGaiaOverlay: boolean;
  showWiseOverlay: boolean;
  showMostOverlay: boolean;
  showFlagsOverlay: boolean;
  checkedObjectIds: string[];
  zoom: number;
  panX: number;
  panY: number;
}

/** Single-epoch detection point along a click-tracked target trajectory. */
export interface TracedWaypoint {
  epoch_id: string;
  epoch_index: number;
  mjd: number;
  iso_date: string;
  x: number;
  y: number;
  ra: number;
  dec: number;
  peak_flux: number;
  snr: number;
  status: WaypointTrackingStatus;
  visible: boolean;
}

/** Multi-epoch trajectory summary for a single click-tracked body branch. */
export interface TracedObjectSummary {
  trackId: string;
  groupId: string;
  branchIndex: number;
  totalBranchesInClick: number;
  isOverlappedAtOrigin: boolean;
  color: string;
  status: TrackSummaryStatus;
  statusMessage: string;
  originEpochIndex: number;
  clickX: number;
  clickY: number;
  originCentroidX: number;
  originCentroidY: number;
  ra: number;
  dec: number;
  x1: number;
  y1: number;
  xN: number;
  yN: number;
  waypoints: TracedWaypoint[];
  lostEpochIds: string[];
  properMotionArcsecPerYr: number;
  positionAngleDeg: number;
  matchedObjectId: string | null;
  matchedCatalog: CatalogSourceType | null;
  matchedLabel: string | null;
  isUncatalogedMover: boolean;
}

/** Aggregate result emitted by `detectAndTrackBodiesAtClick`. */
export interface ClickTrackingResult {
  status: TrackSummaryStatus;
  message: string;
  clickX: number;
  clickY: number;
  originEpochIndex: number;
  isOverlapped: boolean;
  tracks: TracedObjectSummary[];
}

/** Diagnostic metadata recorded after the latest canvas click-to-track attempt. */
export interface ClickDiagnosticInfo {
  clickX: number;
  clickY: number;
  epochIndex: number;
  status: TrackSummaryStatus;
  message: string;
  tracksAdded: number;
}

/** Active 3D night-sky globe camera frustum state. */
export interface SkyViewportFrustum {
  ra: number;
  dec: number;
  fovDeg: number;
  aspect?: number;
}

/** Complete public API exposed by `SkyExplorerProvider` and `useSkyExplorer`. */
export interface SkyExplorerContextValue {
  activePresetSlug: string | null;
  coordinates: { ra: number; dec: number; fovArcmin: number };
  manifest: SkyPatchManifest | null;
  epochBuffers: Float32Array[][] | null;
  flagsBuffers: Uint8Array[] | null;
  loadingState: {
    stage: 'idle' | 'loading_catalogs' | 'loading_cutouts' | 'ready' | 'error';
    message?: string;
    fromClientCache?: boolean;
  };
  renderSettings: ViewportRenderSettings;
  updateRenderSettings: (partial: Partial<ViewportRenderSettings>) => void;
  setViewportZoomPan: (zoom: number, panX: number, panY: number) => void;
  resetViewportZoomPan: () => void;
  knownObjects: KnownCelestialObject[];
  objectHighlighterColors?: Record<string, string>;
  globalKnownObjects?: GlobalKnownObjectEntry[];
  toggleKnownObject: (objectId: string) => void;
  checkAllKnownObjects: () => void;
  uncheckAllKnownObjects: () => void;
  resetDefaultKnownObjects: () => void;
  focusAndTrackKnownObject?: (
    obj: KnownCelestialObject | GlobalKnownObjectEntry,
    targetSlug?: string,
  ) => void;
  playback: {
    isPlaying: boolean;
    isLooping: boolean;
    fps: number;
    play: () => void;
    pause: () => void;
    stepForward: () => void;
    stepBackward: () => void;
    toggleLoop: () => void;
    setFps: (fps: number) => void;
  };
  selectedObject: TracedObjectSummary | null;
  trackedTargets: TracedObjectSummary[];
  lastClickDiagnostic: ClickDiagnosticInfo | null;
  removeTrackedTarget: (trackId: string) => void;
  clearTrackedTargets: () => void;
  activeSpectrum: SpectrumPayload | null;
  isSpectrumLoading: boolean;
  selectPreset: (slug: string, preferredEpochMjd?: number) => void;
  navigateToCoordinates: (
    ra: number,
    dec: number,
    fovArcmin?: number,
    preferredEpochMjd?: number,
  ) => void;
  retryActivePatch?: () => void;
  inspectPixelOrCoord: (x: number, y: number, originEpochIndex?: number) => void;
  activeMissions?: MissionName[];
  filteredEpochIndices?: number[];
  toggleMissionFilter?: (mission: MissionName) => void;
  getSkyFrustum?: () => SkyViewportFrustum;
  isBlinkerModalOpen?: boolean;
  openBlinkerModal?: (target?: string | { ra: number; dec: number; fovArcmin?: number }) => void;
  closeBlinkerModal?: () => void;
  isSpectraModalOpen?: boolean;
  openSelectedSpectrum?: (coords?: { ra: number; dec: number; label?: string }) => Promise<void>;
  closeSpectraModal?: () => void;
  untrackObjectById?: (objectId: string) => void;
  requestGlobeFocus?: (ra: number, dec: number, fovDeg?: number) => void;
  fetchQualityMode?: FetchQualityMode;
  setFetchQualityMode?: (mode: FetchQualityMode) => void;
  activePatchDtype?: BinaryDtype;
  isNetworkSlow?: boolean;
  measuredBandwidthMbps?: number | null;
  refetchActivePatch?: () => void;
  stepAdjacentPatch?: (direction: 'north' | 'south' | 'east' | 'west') => void;
  recentPatches?: ReadonlyArray<RecentPatchEntry>;
  clearRecentPatches?: () => void;
}
