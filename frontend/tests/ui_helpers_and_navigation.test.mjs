import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { readStoredNumber } from '../src/hooks/useResizableSplit.ts';
import { TRACKING_SQUARE_COLORS } from '../src/types/contracts.ts';
import { attachAladinIdleThrottler } from '../src/utils/skyProjection.ts';
import {
  HOVER_DETAILS_DELAY_MS,
  computeSmartCalloutLayout,
} from '../src/utils/calloutPlacement.ts';
import {
  CDS_MIRROR_SPHEREX_COLOR_HIPS_URL,
  CDS_SPHEREX_COLOR_HIPS_URL,
  IRSA_2MASS_HIPS_URL,
  IRSA_SPHEREX_CUBE_HIPS_URL,
  SUPPORTED_GLOBE_PROJECTIONS,
  buildVisibleSkyPanels,
  clampGlobeFov,
  resolveBestDeepSkyHiPS,
  resolveGlobeClickAction,
  resolveGlobeKeyboardAction,
  resolveGlobeLodTier,
  stepPanCoordinates,
  targetFovForClickAction,
} from '../src/utils/globeNavigation.ts';
import {
  WAVELENGTH_PRESETS,
  assignDeterministicHighlighterColors,
  catalogBadgeLabel,
  clearCheckedObjectIds,
  extractAvailableMissions,
  formatMagnitude,
  formatProperMotion,
  isWavelengthPresetEnabled,
  reconcileWavelengthRange,
  resolveWavelengthConfigForMissions,
  fuzzyScoreText,
  fuzzySearchKnownObjects,
  getDefaultCheckedObjectIds,
  inferObjectClass,
  isMovingSolarSystemObject,
  isWavelengthPresetActive,
  objectClassBadgeLabel,
  pickClosestEpochIndex,
  pickDistinctHighlighterColors,
  pickNearestAllowedEpoch,
  resolveFilteredEpochIndices,
  selectAllObjectIds,
  toggleObjectId,
} from '../src/utils/knownObjectSelection.ts';
import { ObservatoryLogger } from '../src/utils/observatoryLogger.ts';
import {
  angularSeparationDeg,
  computeMagneticSnap,
  decodeHygieneFlagMask,
  formatBlinkerSubtitle,
  formatEpochDateShort,
  formatImageProvenanceBadge,
  formatRaDecCompact,
  formatRelativeDataAge,
  getMarkerShapeKind,
  getSurveyArchivalEra,
  isObjectInPatchViewBox,
  isObjectInSkyFrustum,
  parseSkySearchInput,
  resolvePositionFromAnchor,
  sortKnownObjectsCheckedFirst,
} from '../src/utils/observatoryMath.ts';
import { renderEpochBuffersToRgba } from '../src/utils/pixelEngine.ts';
import { replaceUrlState } from '../src/utils/urlStateSync.ts';
import {
  PRESET_SLUGS,
  PRESETS_DIR,
  loadPresetEpochAndFlags,
  readPresetKnownObjects,
} from './helpers/fixtures.mjs';

test('ISS-16: default selection checks notable movers and leaves stationary field stars unchecked', () => {
  assert.equal(PRESET_SLUGS.length, 12);
  for (const slug of PRESET_SLUGS) {
    const known = readPresetKnownObjects(slug);
    const defaults = getDefaultCheckedObjectIds(known);
    assert.ok(defaults.length >= 1, `${slug}: expected at least one notable object checked`);
    for (const obj of known) {
      assert.equal(
        defaults.includes(obj.object_id),
        obj.default_checked,
        `${slug}: ${obj.object_id} default selection must follow default_checked`,
      );
    }
    assert.ok(
      known.some((obj) => !obj.default_checked),
      `${slug}: expected stationary background stars to start unchecked`,
    );
  }
});

test('ISS-23: highlighter color palette has >=16 distinct colors and fuzzy search matches typos and catalog designations', () => {
  assert.ok(TRACKING_SQUARE_COLORS.length >= 16);
  assert.equal(new Set(TRACKING_SQUARE_COLORS).size, TRACKING_SQUARE_COLORS.length);

  // Sampling 16 colors without replacement produces 16 distinct colors
  const sampled = pickDistinctHighlighterColors(16);
  assert.equal(sampled.length, 16);
  assert.equal(new Set(sampled).size, 16);

  // Fuzzy search ranks exact, substring, subsequence, and typo queries accurately
  assert.ok(fuzzyScoreText('barnrd', "Barnard's Star (GJ 699)") >= 0.45);
  assert.ok(fuzzyScoreText('w0855', 'WISE 0855-0714') >= 0.5);

  const barnardKnown = readPresetKnownObjects('barnards_star');
  const matches = fuzzySearchKnownObjects(barnardKnown, 'barnrd');
  assert.ok(matches.length >= 1);
  assert.ok(matches[0].label.includes("Barnard's Star"));
});

test('ISS-16: Check All selects every object, Uncheck All clears to [], Notable Only restores defaults', () => {
  for (const slug of PRESET_SLUGS) {
    const known = readPresetKnownObjects(slug);

    const all = selectAllObjectIds(known);
    assert.deepEqual(
      all,
      known.map((obj) => obj.object_id),
      `${slug}: Check All must include every known object`,
    );

    assert.deepEqual(clearCheckedObjectIds(), [], `${slug}: Uncheck All must yield []`);

    const restored = getDefaultCheckedObjectIds(known);
    assert.notDeepEqual(all, restored, `${slug}: Notable Only must differ from Check All`);
    assert.deepEqual(
      restored,
      known.filter((obj) => obj.default_checked).map((obj) => obj.object_id),
    );
  }
});

test('ISS-16: toggling an object id adds or removes only that id without mutating the input', () => {
  const start = ['a', 'b'];
  const added = toggleObjectId(start, 'c');
  assert.deepEqual(added, ['a', 'b', 'c']);
  assert.deepEqual(start, ['a', 'b'], 'input array must not be mutated');

  const removed = toggleObjectId(added, 'a');
  assert.deepEqual(removed, ['b', 'c']);

  assert.deepEqual(toggleObjectId(toggleObjectId([], 'x'), 'x'), []);
});

test('ISS-16: wavelength quick-filter pills cover the four required ranges and highlight exclusively', () => {
  assert.deepEqual(
    WAVELENGTH_PRESETS.map((p) => p.range),
    [
      [0.75, 5.0],
      [3.8, 5.0],
      [2.4, 3.8],
      [0.75, 2.4],
    ],
  );
  for (const preset of WAVELENGTH_PRESETS) {
    const active = WAVELENGTH_PRESETS.filter((p) =>
      isWavelengthPresetActive(preset.range, p.range),
    );
    assert.deepEqual(
      active.map((p) => p.id),
      [preset.id],
      `only the ${preset.id} pill should be active for its own range`,
    );
    assert.ok(preset.range[0] >= 0.75 && preset.range[1] <= 5.0);
  }
  assert.equal(isWavelengthPresetActive([1.0, 4.0], [0.75, 5.0]), false);
});

test('SkyView, WISE, and SPHEREx clamp knobs on the full 0.65–5.0 µm rail and disable out-of-range pills', () => {
  const skyViewOnly = resolveWavelengthConfigForMissions(
    [{ mission: 'SkyView', count: 2 }],
    ['SPHEREx', 'WISE', 'SkyView'],
    'SkyView',
  );
  assert.equal(skyViewOnly.isMissionRestricted, true);
  assert.equal(skyViewOnly.railMin, 0.65);
  assert.equal(skyViewOnly.railMax, 5.0);
  assert.equal(skyViewOnly.min, 0.65);
  assert.equal(skyViewOnly.max, 0.85);
  assert.deepEqual(skyViewOnly.defaultRange, [0.65, 0.85]);
  assert.equal(
    skyViewOnly.presets.some((p) => isWavelengthPresetEnabled(p.range, skyViewOnly)),
    false,
    'All 4 SPHEREx presets are disabled for SkyView (0.65–0.85 µm)',
  );

  // WISE-only clamps knobs to [2.4, 5.0] and enables Cold + Ice pills only
  const wiseOnly = resolveWavelengthConfigForMissions(
    [{ mission: 'WISE', count: 2 }],
    ['SPHEREx', 'WISE'],
    'WISE',
  );
  assert.equal(wiseOnly.isMissionRestricted, true);
  assert.equal(wiseOnly.railMin, 0.65);
  assert.equal(wiseOnly.railMax, 5.0);
  assert.equal(wiseOnly.min, 2.4);
  assert.equal(wiseOnly.max, 5.0);
  assert.deepEqual(wiseOnly.defaultRange, [2.4, 5.0]);
  assert.deepEqual(
    wiseOnly.presets.filter((p) => isWavelengthPresetEnabled(p.range, wiseOnly)).map((p) => p.id),
    ['cold', 'molecules'],
  );

  // Fallback to primaryMission when availableMissions is empty
  const skyViewPrimaryFallback = resolveWavelengthConfigForMissions([], ['SkyView'], 'SkyView');
  assert.equal(skyViewPrimaryFallback.isMissionRestricted, true);
  assert.deepEqual(skyViewPrimaryFallback.defaultRange, [0.65, 0.85]);

  // SPHEREx or mixed missions retain 0.75–5.0 µm knob bounds on the 0.65–5.0 µm rail
  const spherexConfig = resolveWavelengthConfigForMissions(
    [
      { mission: 'SPHEREx', count: 3 },
      { mission: 'SkyView', count: 2 },
    ],
    ['SPHEREx', 'SkyView'],
    'SPHEREx',
  );
  assert.equal(spherexConfig.isMissionRestricted, false);
  assert.equal(spherexConfig.railMin, 0.65);
  assert.equal(spherexConfig.railMax, 5.0);
  assert.equal(spherexConfig.min, 0.75);
  assert.equal(spherexConfig.max, 5.0);
  assert.equal(
    spherexConfig.presets.every((p) => isWavelengthPresetEnabled(p.range, spherexConfig)),
    true,
  );

  // Reconciling wavelength range across mission transitions resets cleanly
  assert.deepEqual(reconcileWavelengthRange([0.75, 5.0], skyViewOnly), [0.65, 0.85]);
  assert.deepEqual(reconcileWavelengthRange([0.65, 0.85], spherexConfig, skyViewOnly), [0.75, 5.0]);
  assert.deepEqual(reconcileWavelengthRange([0.75, 5.0], wiseOnly, spherexConfig), [2.4, 5.0]);
  assert.deepEqual(reconcileWavelengthRange([3.8, 5.0], spherexConfig), [3.8, 5.0]);
});

test('ISS-16: known-object rows show a catalog badge, magnitude and proper motion for every preset object', () => {
  assert.equal(catalogBadgeLabel('gaia_dr3'), 'Gaia DR3');
  assert.equal(catalogBadgeLabel('wise'), 'WISE');
  assert.equal(catalogBadgeLabel('most'), 'MOST');

  assert.equal(formatMagnitude(16.44), 'mag 16.4');
  assert.equal(formatMagnitude(null), 'mag n/a');
  assert.equal(formatProperMotion(0.013), 'μ 0.013″/yr');
  assert.equal(formatProperMotion(8.12), 'μ 8.12″/yr');
  assert.equal(formatProperMotion(null), 'μ n/a');

  for (const slug of PRESET_SLUGS) {
    for (const obj of readPresetKnownObjects(slug)) {
      const row = [
        catalogBadgeLabel(obj.catalog),
        formatMagnitude(obj.magnitude),
        formatProperMotion(obj.proper_motion_arcsec_yr),
      ].join(' ');
      assert.ok(!/NaN|undefined|null/.test(row), `${slug}/${obj.object_id}: bad row "${row}"`);
    }
  }
});

test('Automated pipeline UI helpers: catalog/class badges, mover detection, and canvas pixel rendering', () => {
  assert.equal(catalogBadgeLabel('simbad'), 'SIMBAD');
  assert.equal(catalogBadgeLabel('jpl_horizons'), 'JPL Horizons');

  const planetObj = {
    object_id: 'ephem:599',
    label: 'Jupiter',
    catalog: 'jpl_horizons',
    object_type: 'solar_system',
    object_class: 'planet',
    ra: 120.0,
    dec: 20.0,
    x: 128,
    y: 128,
    magnitude: -2.3,
    proper_motion_arcsec_yr: null,
    w1_w2_color: null,
    default_checked: true,
    epoch_positions: [
      { epoch_id: 'pass_1', mjd: 60800, ra: 120.0, dec: 20.0, x: 110, y: 120 },
      { epoch_id: 'pass_2', mjd: 60900, ra: 120.02, dec: 20.01, x: 145, y: 132 },
    ],
  };
  assert.equal(inferObjectClass(planetObj), 'planet');
  assert.equal(objectClassBadgeLabel('planet'), 'Planet');
  assert.equal(isMovingSolarSystemObject(planetObj), true);

  const bdObj = {
    ...planetObj,
    object_id: 'simbad:WISE_0855',
    catalog: 'simbad',
    object_type: 'brown_dwarf_candidate',
    object_class: null,
    epoch_positions: [{ epoch_id: 'pass_1', mjd: 60800, ra: 133.7, dec: -7.2, x: 128, y: 128 }],
  };
  assert.equal(inferObjectClass(bdObj), 'brown_dwarf');
  assert.equal(objectClassBadgeLabel('brown_dwarf'), 'Brown Dwarf');

  // Render 6-detector preset pass_1 + pass_2 buffers to 256x256 RGBA bytes
  const { epochBuffers } = loadPresetEpochAndFlags('wise_0855');

  const blinkRgba = renderEpochBuffersToRgba({
    epochBuffers,
    activeEpochIndex: 0,
    comparisonMode: 'blink',
    contrastStretch: 'asinh',
    wavelengthRangeUm: [0.75, 5.0],
  });
  assert.equal(blinkRgba.length, 256 * 256 * 4);
  assert.equal(blinkRgba[3], 255);

  const diffRgba = renderEpochBuffersToRgba({
    epochBuffers,
    activeEpochIndex: 1,
    comparisonMode: 'subtract',
    contrastStretch: 'zscale',
    wavelengthRangeUm: [3.8, 5.0],
  });
  assert.equal(diffRgba.length, 256 * 256 * 4);
});

test('Redesign math helpers enforce frustum filtering, deferred sorting, magnetic snap, hygiene flags, and 3-shape markers', () => {
  // Frustum filtering excludes far-hemisphere objects while keeping visible region targets
  const wiseFrustum = { ra: 133.7951, dec: -7.2451, fovDeg: 65, aspect: 1.4 };
  assert.equal(isObjectInSkyFrustum(133.8, -7.2, wiseFrustum), true);
  assert.equal(isObjectInSkyFrustum(313.8, 7.2, wiseFrustum), false);
  assert.ok(angularSeparationDeg(0, 0, 180, 0) > 179.9);

  // Patch viewBox filtering honors zoom/pan bounds
  assert.equal(
    isObjectInPatchViewBox(128, 128, { minX: 96, minY: 96, viewWidth: 64, viewHeight: 64 }),
    true,
  );
  assert.equal(
    isObjectInPatchViewBox(12, 12, { minX: 96, minY: 96, viewWidth: 64, viewHeight: 64 }),
    false,
  );

  // Deferred sort moves checked items to top while preserving relative group order
  const rows = [{ object_id: 'a' }, { object_id: 'b' }, { object_id: 'c' }, { object_id: 'd' }];
  assert.deepEqual(
    sortKnownObjectsCheckedFirst(rows, ['c', 'a']).map((r) => r.object_id),
    ['a', 'c', 'b', 'd'],
  );

  // Magnetic snapping clamps safe 16px margins and detects corners within 24px
  const snapped = computeMagneticSnap(
    { x: 30, y: 20 },
    { width: 291, height: 320 },
    { width: 1440, height: 1024 },
  );
  assert.equal(snapped.x, 16);
  assert.equal(snapped.y, 16);
  assert.equal(snapped.edgeAnchor, 'top-left');

  const resolvedRight = resolvePositionFromAnchor(
    { x: 1000, y: 16, edgeAnchor: 'top-right' },
    { width: 291, height: 320 },
    { width: 1280, height: 800 },
  );
  assert.equal(resolvedRight.x, 1280 - 291 - 16);
  assert.equal(resolvedRight.y, 16);

  // Hygiene flag mask decoding identifies all contamination bits
  assert.equal(decodeHygieneFlagMask(0).isFlagged, false);
  const decodedCrSat = decodeHygieneFlagMask(1 | 2 | 4);
  assert.equal(decodedCrSat.isFlagged, true);
  assert.equal(decodedCrSat.severity, 'critical');
  assert.equal(decodedCrSat.bits.length, 3);

  // Search input parses decimal, HMS/DMS sexagesimal, and object names
  const decSearch = parseSkySearchInput('133.7951, -7.2451');
  assert.equal(decSearch.kind, 'coords');
  if (decSearch.kind === 'coords') {
    assert.ok(Math.abs(decSearch.ra - 133.7951) < 1e-4);
    assert.ok(Math.abs(decSearch.dec - -7.2451) < 1e-4);
  }
  const sexaSearch = parseSkySearchInput('08:55:10.8 -07:14:42');
  assert.equal(sexaSearch.kind, 'coords');
  assert.equal(parseSkySearchInput('Jupiter').kind, 'text');

  // Strictly 3 hollow shapes: square when tracked, diamond for moving Solar System, triangle for star/deep-sky
  assert.equal(
    getMarkerShapeKind(
      { object_class: 'star', object_type: 'star', label: 'Sirius', epoch_positions: [] },
      false,
    ),
    'triangle',
  );
  assert.equal(
    getMarkerShapeKind(
      {
        object_class: 'comet',
        object_type: 'solar_system',
        label: '3I/ATLAS',
        epoch_positions: [],
      },
      false,
    ),
    'diamond',
  );
  assert.equal(
    getMarkerShapeKind(
      { object_class: 'star', object_type: 'star', label: 'Sirius', epoch_positions: [] },
      true,
    ),
    'square',
  );

  // Expanded global_known_objects.json contains planets, moons, comets, asteroids, stars, nebulae, and galaxies
  const globalCatalog = JSON.parse(
    fs.readFileSync(path.join(PRESETS_DIR, 'global_known_objects.json'), 'utf8'),
  );
  assert.ok(globalCatalog.length >= 620);
  const classes = new Set(globalCatalog.map((item) => item.object_class || item.object_type));
  for (const requiredClass of ['planet', 'moon', 'comet', 'asteroid', 'nebula', 'galaxy']) {
    assert.ok(classes.has(requiredClass), `Missing expanded class ${requiredClass}`);
  }
});

test('Progressive globe click navigation reveals sky panels without auto-opening on zoom', () => {
  assert.equal(resolveGlobeClickAction(70), 'zoom_medium');
  assert.equal(targetFovForClickAction('zoom_medium'), 8.0);
  assert.equal(resolveGlobeClickAction(8.0), 'zoom_panels');
  assert.equal(targetFovForClickAction('zoom_panels'), 1.6);
  assert.equal(resolveGlobeClickAction(1.6), 'open_panel');

  assert.equal(buildVisibleSkyPanels(133.79, -7.25, 8.0).length, 0);
  const panels = buildVisibleSkyPanels(133.79, -7.25, 1.6);
  assert.ok(panels.length > 0);
  assert.ok(panels.some((cell) => cell.key === '133.80_-7.25'));

  const pannedNorth = stepPanCoordinates(359.5, 88.0, 20, 'north');
  assert.ok(pannedNorth.dec <= 89.9);
  const pannedEast = stepPanCoordinates(359.0, 0, 20, 'east');
  assert.ok(pannedEast.ra >= 0 && pannedEast.ra < 360);
  // North-up orientation: East (screen-left) increases RA, West (screen-right) decreases RA
  const fromCenterEast = stepPanCoordinates(180.0, 0, 20, 'east');
  const fromCenterWest = stepPanCoordinates(180.0, 0, 20, 'west');
  assert.ok(fromCenterEast.ra > 180.0);
  assert.ok(fromCenterWest.ra < 180.0);
  const fineEast = stepPanCoordinates(180.0, 0, 20, 'east', 0.35);
  assert.ok(fineEast.ra - 180.0 < (fromCenterEast.ra - 180.0) * 0.5);

  assert.deepEqual(resolveGlobeKeyboardAction({ key: 'ArrowLeft' }), {
    type: 'pan',
    dir: 'east',
    fine: false,
  });
  assert.deepEqual(resolveGlobeKeyboardAction({ key: 'd', shiftKey: true }), {
    type: 'pan',
    dir: 'west',
    fine: true,
  });
  assert.deepEqual(resolveGlobeKeyboardAction({ key: '=' }), { type: 'zoom', factor: 0.72 });
  assert.deepEqual(resolveGlobeKeyboardAction({ key: '-' }), { type: 'zoom', factor: 1.38 });
  assert.deepEqual(resolveGlobeKeyboardAction({ key: '0' }), { type: 'zoom', factor: 'reset' });
  assert.equal(resolveGlobeKeyboardAction({ key: '+', ctrlKey: true }), null);
  assert.equal(resolveGlobeKeyboardAction({ key: 'ArrowUp', metaKey: true }), null);
  assert.equal(resolveGlobeKeyboardAction({ key: 'x' }), null);

  assert.equal(clampGlobeFov(999), 160);
  assert.equal(clampGlobeFov(0.01), 0.25);
});

test('Smart callout placement flips away from right edge & UI panels without occluding object', () => {
  assert.ok(HOVER_DETAILS_DELAY_MS >= 700 && HOVER_DETAILS_DELAY_MS <= 1500);

  // Target near right edge must flip card to the left so marker remains uncovered
  const rightTargetLayout = computeSmartCalloutLayout({
    markerStageX: 890,
    markerStageY: 310,
    stageW: 960,
    stageH: 640,
    cardW: 316,
    cardH: 186,
  });
  assert.equal(rightTargetLayout.side, 'left');
  assert.ok(rightTargetLayout.cardLeft + 316 < 890);

  // Target overlapping top-left controls box should avoid avoidRects cleanly
  const avoidControlsLayout = computeSmartCalloutLayout({
    markerStageX: 355,
    markerStageY: 160,
    stageW: 1100,
    stageH: 700,
    cardW: 316,
    cardH: 186,
    avoidRects: [{ left: 14, top: 14, right: 310, bottom: 420 }],
  });
  assert.equal(avoidControlsLayout.side, 'right');
  assert.ok(avoidControlsLayout.cardLeft > 355);
});

test('Coordinate labels use full names and image provenance distinguishes primary, extension & simulated', () => {
  const coordText = formatRaDecCompact(133.7951, -7.2451);
  assert.ok(coordText.startsWith('Right Ascension 133.7951°, Declination -7.2451°'));
  assert.equal(coordText.includes('—'), false);

  const primaryBadge = formatImageProvenanceBadge({
    primary_mission: 'SPHEREx',
    has_multi_epoch_spherex: true,
    uses_wise_fallback: false,
    baseline_days: 180,
    status_note: 'SPHEREx',
  });
  assert.equal(primaryBadge.tier, 'primary');

  const extBadge = formatImageProvenanceBadge({
    primary_mission: 'SkyView',
    has_multi_epoch_spherex: false,
    uses_wise_fallback: false,
    baseline_days: 0,
    status_note: 'DSS2 IR',
    provenance_tier: 'extension',
    survey_name: 'NASA SkyView (DSS2 IR / Red)',
  });
  assert.equal(extBadge.tier, 'extension');
  assert.equal(extBadge.label, 'NASA SkyView (DSS2 IR / Red)');

  const simBadge = formatImageProvenanceBadge({
    primary_mission: 'Simulated',
    has_multi_epoch_spherex: false,
    uses_wise_fallback: false,
    baseline_days: 0,
    status_note: 'Synthetic baseline',
  });
  assert.equal(simBadge.tier, 'simulated');
});

test('ObservatoryLogger enforces configurable levels and FIFO bufferSize, and replaceUrlState deduplicates', () => {
  const logger = new ObservatoryLogger({ level: 'warn', bufferSize: 20 });
  assert.equal(logger.shouldLog('debug'), false);
  assert.equal(logger.shouldLog('info'), false);
  assert.equal(logger.info('test', 'ignored_info'), null);
  assert.ok(logger.warn('test', 'recorded_warn', { ok: true }));

  logger.configure({ level: 'debug', bufferSize: 20 });
  assert.equal(logger.shouldLog('debug'), true);
  for (let i = 0; i < 25; i += 1) {
    logger.debug('test', `evt_${i}`);
  }
  const entries = logger.getEntries();
  assert.equal(entries.length, 20);
  assert.equal(entries[entries.length - 1].event, 'evt_24');

  let calls = 0;
  const mockLocation = { pathname: '/', search: '', hash: '' };
  const mockHistory = {
    replaceState(_state, _unused, url) {
      calls += 1;
      const qIdx = url.indexOf('?');
      mockLocation.search = qIdx >= 0 ? url.slice(qIdx) : '';
    },
  };
  const firstUrl = replaceUrlState(
    { target: 'wise0855', fov: 70 },
    { location: mockLocation, history: mockHistory },
  );
  replaceUrlState(
    { target: 'wise0855', fov: 70 },
    { location: mockLocation, history: mockHistory },
  );
  assert.ok(firstUrl.includes('target=wise0855'));
  assert.equal(calls, 1);
});

test('Resilient HiPS hierarchy, LoD tiers, projections, and mission epoch filters', async () => {
  assert.equal(resolveGlobeLodTier(45), 'allwise');
  assert.equal(resolveGlobeLodTier(8), '2mass');
  assert.equal(resolveGlobeLodTier(2.5), 'spherex_deep');
  assert.ok(SUPPORTED_GLOBE_PROJECTIONS.some((p) => p.id === 'SIN'));
  assert.ok(SUPPORTED_GLOBE_PROJECTIONS.some((p) => p.id === 'AIT'));

  const primaryCds = await resolveBestDeepSkyHiPS(async (url) =>
    url.startsWith(CDS_SPHEREX_COLOR_HIPS_URL),
  );
  assert.equal(primaryCds.source, 'cds_qr2');
  assert.equal(primaryCds.label, 'SPHEREx QR2 Color (CDS)');

  const mirrorCds = await resolveBestDeepSkyHiPS(async (url) =>
    url.startsWith(CDS_MIRROR_SPHEREX_COLOR_HIPS_URL),
  );
  assert.equal(mirrorCds.source, 'cds_qr2_mirror');
  assert.equal(mirrorCds.label, 'SPHEREx QR2 Color (Mirror)');

  const irsaCube = await resolveBestDeepSkyHiPS(async (url) =>
    url.startsWith(IRSA_SPHEREX_CUBE_HIPS_URL),
  );
  assert.equal(irsaCube.source, 'irsa_spherex_cube');

  const safeFallback = await resolveBestDeepSkyHiPS(async () => false);
  assert.equal(safeFallback.source, 'irsa_2mass_fallback');
  assert.equal(safeFallback.url, IRSA_2MASS_HIPS_URL);

  const epochs = [
    { mission: 'SPHEREx' },
    { mission: 'WISE' },
    { mission: 'SPHEREx' },
    { mission: 'SkyView' },
  ];
  assert.deepEqual(extractAvailableMissions(epochs), [
    { mission: 'SPHEREx', count: 2 },
    { mission: 'WISE', count: 1 },
    { mission: 'SkyView', count: 1 },
  ]);
  const onlyWise = resolveFilteredEpochIndices(epochs, ['WISE']);
  assert.deepEqual(onlyWise, [1]);
  assert.equal(pickNearestAllowedEpoch(2, [0, 3]), 3);
  assert.equal(pickNearestAllowedEpoch(3, [0, 3]), 3);
});

test('Phase 1 assignDeterministicHighlighterColors is deterministic and preserves Stage A -> Stage B colors', () => {
  const stageA = [{ object_id: 'gaia:101' }, { object_id: 'wise:202' }, { object_id: 'most:303' }];
  const firstRun = assignDeterministicHighlighterColors(stageA);
  const secondRun = assignDeterministicHighlighterColors(stageA);
  assert.deepEqual(firstRun, secondRun);
  assert.equal(new Set(Object.values(firstRun)).size, 3);

  // Stage B reorders objects and adds new ones; Stage A object_id -> color mappings stay intact
  const stageB = [
    { object_id: 'simbad:999' },
    { object_id: 'most:303' },
    { object_id: 'gaia:101' },
    { object_id: 'wise:202' },
    { object_id: 'gaia:404' },
  ];
  const stageBColors = assignDeterministicHighlighterColors(stageB, firstRun);
  assert.equal(stageBColors['gaia:101'], firstRun['gaia:101']);
  assert.equal(stageBColors['wise:202'], firstRun['wise:202']);
  assert.equal(stageBColors['most:303'], firstRun['most:303']);
  assert.equal(new Set(Object.values(stageBColors)).size, 5);
  assert.deepEqual(assignDeterministicHighlighterColors([]), {});

  // Verify full 20-color canonical palette coverage including indices 8..19
  const twentyObjects = Array.from({ length: 20 }, (_, idx) => ({ object_id: `obj:${idx}` }));
  const twentyColors = assignDeterministicHighlighterColors(twentyObjects);
  assert.equal(new Set(Object.values(twentyColors)).size, 20);
  for (const hex of Object.values(twentyColors)) {
    assert.ok(TRACKING_SQUARE_COLORS.includes(hex));
  }
  assert.equal(TRACKING_SQUARE_COLORS[8], '#AACC00');
  assert.equal(TRACKING_SQUARE_COLORS[19], '#FEE440');
});

test('Phase 2 readStoredNumber safely validates localStorage bounds and SSR fallback', () => {
  // In Node SSR (window undefined), returns fallback
  assert.equal(readStoredNumber('sh_test_width', 292, 240, 420), 292);

  const origWindowDesc = Object.getOwnPropertyDescriptor(globalThis, 'window');
  const storageMap = new Map();
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      localStorage: {
        getItem: (k) => (storageMap.has(k) ? storageMap.get(k) : null),
      },
    },
  });
  try {
    // Missing key or empty string -> fallback even when 0 is within [min, max]
    assert.equal(readStoredNumber('missing', 292, 240, 420), 292);
    assert.equal(readStoredNumber('missing', 0.5, 0, 1), 0.5);
    storageMap.set('empty', '');
    storageMap.set('whitespace', '   ');
    assert.equal(readStoredNumber('empty', 0.5, 0, 1), 0.5);
    assert.equal(readStoredNumber('whitespace', 0.5, 0, 1), 0.5);
    // Out-of-bounds / non-numeric -> fallback
    storageMap.set('too_small', '120');
    storageMap.set('too_large', '999');
    storageMap.set('nan_val', 'abc');
    assert.equal(readStoredNumber('too_small', 292, 240, 420), 292);
    assert.equal(readStoredNumber('too_large', 292, 240, 420), 292);
    assert.equal(readStoredNumber('nan_val', 292, 240, 420), 292);
    assert.equal(readStoredNumber('empty', 292, 240, 420), 292);
    // Valid stored number (including 0 when inside [min, max]) -> parsed value
    storageMap.set('valid', '340');
    storageMap.set('zero', '0');
    assert.equal(readStoredNumber('valid', 292, 240, 420), 340);
    assert.equal(readStoredNumber('zero', 0.5, 0, 1), 0);
  } finally {
    if (origWindowDesc) {
      Object.defineProperty(globalThis, 'window', origWindowDesc);
    } else {
      delete globalThis.window;
    }
  }
});

test('attachAladinIdleThrottler is event-driven on tile rendering/motion, pauses when hidden, and stops on destroy', () => {
  let redrawCount = 0;
  let renderingTiles = true;
  const scheduledTimers = new Map();
  let nextTimerId = 1;
  const scheduledFrames = [];

  const view = {
    moving: false,
    needRedraw: false,
    wasm: {
      isRendering: () => renderingTiles,
      isInerting: () => false,
    },
    redrawClbk() {
      redrawCount += 1;
    },
  };
  const instance = { gotoRaDec() {}, view };

  const throttler = attachAladinIdleThrottler(instance, {
    idleIntervalMs: 500,
    quietFramesBeforeIdle: 2,
    scheduleFrame: (cb) => scheduledFrames.push(cb),
    scheduleTimer: (cb) => {
      const id = nextTimerId++;
      scheduledTimers.set(id, cb);
      return id;
    },
    cancelTimer: (id) => {
      scheduledTimers.delete(id);
    },
  });

  // While HiPS tiles are actively rendering: runs immediately at 60 FPS
  view.redrawClbk();
  view.redrawClbk();
  assert.equal(redrawCount, 2);
  assert.equal(scheduledTimers.size, 0);

  // Once tiles finish rendering: 1 quiet frame runs, 2nd quiet frame drops to idle timer
  renderingTiles = false;
  view.redrawClbk();
  assert.equal(redrawCount, 3);
  assert.equal(scheduledTimers.size, 0);
  view.redrawClbk();
  assert.equal(redrawCount, 3);
  assert.equal(scheduledTimers.size, 1);

  // If a late HiPS tile arrives during an idle keepalive check, ramps back up to 60 FPS
  renderingTiles = true;
  const [idleCb] = [...scheduledTimers.values()];
  scheduledTimers.clear();
  idleCb();
  assert.equal(redrawCount, 4);
  view.redrawClbk();
  assert.equal(redrawCount, 5);
  assert.equal(scheduledTimers.size, 0);

  // Drop back to idle, then wake() cancels timer and schedules immediate frame
  renderingTiles = false;
  view.redrawClbk();
  view.redrawClbk();
  assert.equal(scheduledTimers.size, 1);
  throttler.wake();
  assert.equal(scheduledTimers.size, 0);
  assert.equal(scheduledFrames.length, 1);
  scheduledFrames.shift()();
  assert.equal(redrawCount, 7);

  // Pausing (modal open / tab hidden) halts rendering to 0 FPS until unpaused
  throttler.setPaused(true);
  view.redrawClbk();
  assert.equal(redrawCount, 7);
  assert.equal(scheduledTimers.size, 0);

  // Unpausing schedules immediate resume
  throttler.setPaused(false);
  assert.equal(scheduledFrames.length, 1);
  scheduledFrames.shift()();
  assert.equal(redrawCount, 8);

  // destroy() permanently halts loop
  throttler.destroy();
  view.redrawClbk();
  assert.equal(redrawCount, 8);
});

test('attachAladinIdleThrottler forwards finite rAF timestamps and recovers from transient redraw errors', () => {
  const receivedTimestamps = [];
  const scheduledTimers = new Map();
  const scheduledFrames = [];
  let nextTimerId = 1;
  let fakeNow = 1000;
  let shouldThrow = false;

  const view = {
    moving: false,
    needRedraw: false,
    wasm: {
      isRendering: () => false,
      isInerting: () => false,
    },
    redrawClbk(ts) {
      if (shouldThrow) {
        shouldThrow = false;
        throw new Error('transient wasm error');
      }
      receivedTimestamps.push(ts);
    },
  };
  const instance = { gotoRaDec() {}, view };

  const throttler = attachAladinIdleThrottler(instance, {
    idleIntervalMs: 500,
    quietFramesBeforeIdle: 2,
    now: () => fakeNow,
    scheduleFrame: (cb) => scheduledFrames.push(cb),
    scheduleTimer: (cb) => {
      const id = nextTimerId++;
      scheduledTimers.set(id, cb);
      return id;
    },
    cancelTimer: (id) => {
      scheduledTimers.delete(id);
    },
  });

  // rAF timestamp is forwarded directly so Aladin dt = ts - prevTime never becomes NaN
  view.redrawClbk(16.67);
  view.redrawClbk(33.34);
  assert.deepEqual(receivedTimestamps, [16.67, 33.34]);

  // Idle timer tick supplies a finite fallback timestamp
  view.redrawClbk(50.0);
  assert.equal(scheduledTimers.size, 1);
  fakeNow = 550.0;
  const [idleCb] = [...scheduledTimers.values()];
  scheduledTimers.clear();
  idleCb();
  assert.deepEqual(receivedTimestamps, [16.67, 33.34, 550.0]);

  // Transient redraw throw schedules recovery instead of permanently killing the loop
  shouldThrow = true;
  throttler.wake();
  assert.equal(scheduledTimers.size, 0);
  view.redrawClbk(566.67);
  assert.equal(scheduledTimers.size, 1);

  // Calling wake() while loopBroken recovers immediately onto the next frame
  throttler.wake();
  assert.equal(scheduledFrames.length, 1);
  scheduledFrames.shift()(583.34);
  assert.deepEqual(receivedTimestamps, [16.67, 33.34, 550.0, 583.34]);
});

test('Epoch timestamps, relative age, archival survey eras, closest epoch picker, and coadd subtitle dates', () => {
  assert.equal(formatEpochDateShort('2025-05-06T02:33:29Z'), '2025-05-06');
  assert.equal(formatEpochDateShort(null), 'n/a');

  const fixedNow = Date.parse('2026-11-20T00:00:00Z');
  assert.equal(formatRelativeDataAge('2026-11-20T00:00:00Z', fixedNow), 'today');
  assert.equal(formatRelativeDataAge('2026-11-10T00:00:00Z', fixedNow), '10 d ago');
  assert.equal(formatRelativeDataAge('2026-05-20T00:00:00Z', fixedNow), '6 mo ago');
  assert.equal(formatRelativeDataAge('2025-05-06T00:00:00Z', fixedNow), '1.5 yr ago');

  assert.equal(getSurveyArchivalEra('AllWISE Wide (IRSA)'), '2010–2011 archive');
  assert.equal(getSurveyArchivalEra('2MASS High-Res (IRSA)'), '1997–2001 archive');
  assert.equal(getSurveyArchivalEra('SPHEREx QR2 Color (CDS)'), '2025–2026 archive');
  assert.equal(getSurveyArchivalEra('Unknown Sky Survey'), 'Archival mosaic');

  const sampleEpochs = [
    { mjd: 60801.1, iso_date: '2025-05-06T02:33:29Z', mission: 'SPHEREx' },
    { mjd: 60987.2, iso_date: '2025-11-08T04:31:52Z', mission: 'SPHEREx' },
    { mjd: 61171.1, iso_date: '2026-05-11T02:18:44Z', mission: 'SPHEREx' },
  ];
  assert.equal(pickClosestEpochIndex(sampleEpochs, 60990.0), 1);
  assert.equal(pickClosestEpochIndex(sampleEpochs, 61170.0), 2);
  assert.equal(pickClosestEpochIndex([], 60990.0), 0);

  const coaddSubtitle = formatBlinkerSubtitle({
    manifest: {
      epochs: sampleEpochs,
    },
    renderSettings: {
      mode: 'coadd',
      activeEpochIndex: 0,
      passAIndex: 0,
      passBIndex: 2,
      coaddIndices: [0, 1, 2],
    },
    provenanceShortLabel: 'SPHEREx (Primary)',
  });
  assert.ok(coaddSubtitle.includes('2025-05-06 to 2026-05-11'));
  assert.ok(coaddSubtitle.includes('SPHEREx (Primary)'));
});
