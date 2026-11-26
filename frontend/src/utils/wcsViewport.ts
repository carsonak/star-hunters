/**
 * WCS coordinate transformations, viewport pan/zoom clamping, wavelength Dim & Pop
 * detector weighting, and unified 0.05-degree spatial grid quantization helpers.
 *
 * Kept React-free so Node 24's `node --test` executes this module directly.
 */

import {
  DETECTOR_WAVELENGTH_BOUNDS_UM,
  GRID_HEIGHT,
  GRID_WIDTH,
  type WcsHeader,
} from '../types/contracts.ts';
import { wrapRaDeg } from './observatoryMath.ts';

/**
 * Quantize sky coordinates `(raDeg, decDeg)` onto the canonical spatial cache grid
 * (`0.05°` step by default), wrapping Right Ascension into `[0, 360)` and clamping
 * Declination to `[-90, 90]`.
 *
 * @param raDeg - Right Ascension in decimal degrees.
 * @param decDeg - Declination in decimal degrees.
 * @param gridStepDeg - Quantization step in degrees (defaults to `0.05`).
 * @returns Quantized `{ raQ, decQ }` center coordinates in degrees.
 */
export function quantizeGridCenter(
  raDeg: number,
  decDeg: number,
  gridStepDeg = 0.05,
): { raQ: number; decQ: number } {
  const normRa = wrapRaDeg(raDeg);
  const clampedDec = Math.max(-90, Math.min(90, decDeg));
  const raQ = wrapRaDeg(Number((Math.round(normRa / gridStepDeg) * gridStepDeg).toFixed(2)));
  const decQ = Number((Math.round(clampedDec / gridStepDeg) * gridStepDeg).toFixed(2));
  return { raQ, decQ };
}

/**
 * Compute the quantized `0.05°` spatial grid key matching the backend `compute_grid_key`.
 *
 * @param ra - Right Ascension in decimal degrees.
 * @param dec - Declination in decimal degrees.
 * @param fovArcmin - Field-of-view width in arcminutes (defaults to `5.0`).
 * @param gridStepDeg - Quantization step in degrees (defaults to `0.05`).
 * @returns Canonical cache key formatted as `ra{raQ}_dec{decQ}_fov{fovQ}`.
 */
export function computeGridKey(
  ra: number,
  dec: number,
  fovArcmin: number = 5.0,
  gridStepDeg: number = 0.05,
): string {
  const { raQ, decQ } = quantizeGridCenter(ra, dec, gridStepDeg);
  const fovQ = Number(fovArcmin.toFixed(1));
  return `ra${raQ.toFixed(2)}_dec${decQ.toFixed(2)}_fov${fovQ.toFixed(1)}`;
}

/**
 * Clamp interactive viewport zoom (`1.0..8.0`) and pan offsets (in 256x256 image pixels)
 * so the visible window always stays within `[0, 256] x [0, 256]`.
 *
 * @param zoom - Requested zoom factor.
 * @param panX - Requested horizontal pan offset in image pixels.
 * @param panY - Requested vertical pan offset in image pixels.
 * @returns Clamped `{ zoom, panX, panY }` values.
 */
export function clampViewportPanZoom(
  zoom: number,
  panX: number,
  panY: number,
): { zoom: number; panX: number; panY: number } {
  const safeZoom = Number.isFinite(zoom) ? Math.max(1.0, Math.min(8.0, zoom)) : 1.0;
  const maxPan = Math.max(0, (GRID_WIDTH - GRID_WIDTH / safeZoom) / 2);
  const safePanX = Number.isFinite(panX) ? Math.max(-maxPan, Math.min(maxPan, panX)) : 0;
  const safePanY = Number.isFinite(panY) ? Math.max(-maxPan, Math.min(maxPan, panY)) : 0;
  return {
    zoom: Number(safeZoom.toFixed(3)),
    panX: Number(safePanX.toFixed(2)),
    panY: Number(safePanY.toFixed(2)),
  };
}

/**
 * Compute the active visible sub-region in 256x256 image coordinates for `(zoom, panX, panY)`.
 *
 * @param zoom - Active zoom factor (`1.0..8.0`).
 * @param panX - Horizontal pan offset in image pixels.
 * @param panY - Vertical pan offset in image pixels.
 * @returns Bounding box `{ minX, minY, viewWidth, viewHeight, svgViewBox }`.
 */
export function computeViewportViewBox(
  zoom: number = 1.0,
  panX: number = 0.0,
  panY: number = 0.0,
): {
  minX: number;
  minY: number;
  viewWidth: number;
  viewHeight: number;
  svgViewBox: string;
} {
  const clamped = clampViewportPanZoom(zoom, panX, panY);
  const viewWidth = GRID_WIDTH / clamped.zoom;
  const viewHeight = GRID_HEIGHT / clamped.zoom;
  const minX = GRID_WIDTH / 2 + clamped.panX - viewWidth / 2;
  const minY = GRID_HEIGHT / 2 + clamped.panY - viewHeight / 2;
  return {
    minX: Number(minX.toFixed(3)),
    minY: Number(minY.toFixed(3)),
    viewWidth: Number(viewWidth.toFixed(3)),
    viewHeight: Number(viewHeight.toFixed(3)),
    svgViewBox: `${minX.toFixed(3)} ${minY.toFixed(3)} ${viewWidth.toFixed(3)} ${viewHeight.toFixed(3)}`,
  };
}

/**
 * Inverse-transform a pointer click `(clientX, clientY)` on a zoomed/panned viewport
 * element back into `(0..256, 0..256)` sky-patch image coordinates.
 *
 * @param clientX - Pointer client X coordinate in CSS pixels.
 * @param clientY - Pointer client Y coordinate in CSS pixels.
 * @param rect - Bounding box of the rendered viewport element.
 * @param zoom - Active viewport zoom factor.
 * @param panX - Active horizontal pan offset in image pixels.
 * @param panY - Active vertical pan offset in image pixels.
 * @returns Image-space `{ x, y, inBounds }` coordinates.
 */
export function screenToImageCoordinates(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  zoom: number = 1.0,
  panX: number = 0.0,
  panY: number = 0.0,
): { x: number; y: number; inBounds: boolean } {
  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0, inBounds: false };
  }
  const u = (clientX - rect.left) / rect.width;
  const v = (clientY - rect.top) / rect.height;
  const { minX, minY, viewWidth, viewHeight } = computeViewportViewBox(zoom, panX, panY);
  const x = Number((minX + u * viewWidth).toFixed(2));
  const y = Number((minY + v * viewHeight).toFixed(2));
  const inBounds =
    u >= 0 && u <= 1 && v >= 0 && v <= 1 && x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT;
  return { x, y, inBounds };
}

/**
 * Convert 256x256 image pixel `(x, y)` to ICRS sky `(ra, dec)` using the patch's TAN WCS header.
 *
 * @param x - Horizontal image pixel coordinate.
 * @param y - Vertical image pixel coordinate.
 * @param wcs - Simplified TAN WCS header from the patch manifest.
 * @returns Sky coordinates `{ ra, dec }` rounded to 5 decimal places.
 */
export function pixelToSkyWcs(x: number, y: number, wcs: WcsHeader): { ra: number; dec: number } {
  const [ra0, dec0] = wcs.crval;
  const [x0, y0] = wcs.crpix;
  const [cdeltX, cdeltY] = wcs.cdelt;
  const cosDec = Math.max(0.01, Math.cos((dec0 * Math.PI) / 180.0));
  const ra = (((ra0 + ((x - x0) * cdeltX) / cosDec) % 360.0) + 360.0) % 360.0;
  const dec = Math.max(-90.0, Math.min(90.0, dec0 + (y - y0) * cdeltY));
  return { ra: Number(ra.toFixed(5)), dec: Number(dec.toFixed(5)) };
}

/**
 * Compute per-detector Dim & Pop weights for the active wavelength slice `[minUm, maxUm]`.
 *
 * Detectors outside the slice retain a 5% context floor (`0.05`) so faint reference
 * stars remain faintly legible while cold or hot features pop.
 *
 * @param wavelengthRangeUm - Inclusive `[minUm, maxUm]` filter in micrometres.
 * @returns Six normalized detector weights in `[0.05, 1.0]`.
 */
export function computeDetectorWeights(wavelengthRangeUm: [number, number]): number[] {
  const [minUm, maxUm] = wavelengthRangeUm;
  // When in the optical/near-IR SkyView range (< 0.75 µm), keep Detectors 1–2
  // (where SkyView plates are packed) proportional to the [0.65, 0.85] µm overlap.
  const svRatio =
    minUm < 0.75 ? Math.max(0, Math.min(maxUm, 0.85) - Math.max(minUm, 0.65)) / 0.2 : 0;
  return DETECTOR_WAVELENGTH_BOUNDS_UM.map(([dMin, dMax], idx) => {
    const overlap = Math.max(0, Math.min(maxUm, dMax) - Math.max(minUm, dMin));
    const width = dMax - dMin;
    const ratio = width > 0 ? overlap / width : 0;
    const effectiveRatio = idx <= 1 && minUm < 0.75 ? Math.min(1, Math.max(ratio, svRatio)) : ratio;
    return 0.05 + 0.95 * effectiveRatio;
  });
}

/**
 * Step a sky-patch center to an adjacent quantized `0.05°` tile in North-Up / East-Left
 * ICRS coordinates.
 *
 * @param ra - Current Right Ascension in degrees.
 * @param dec - Current Declination in degrees.
 * @param direction - Cardinal direction to step (`'north' | 'south' | 'east' | 'west'`).
 * @param stepDeg - Grid step in degrees (defaults to `0.05`).
 * @returns Quantized `{ raQ, decQ }` center for the adjacent patch.
 */
export function stepAdjacentPatchCoordinates(
  ra: number,
  dec: number,
  direction: 'north' | 'south' | 'east' | 'west',
  stepDeg = 0.05,
): { raQ: number; decQ: number } {
  let nextRa = wrapRaDeg(ra);
  let nextDec = Math.max(-90, Math.min(90, dec));
  const cosDec = Math.max(0.05, Math.cos((nextDec * Math.PI) / 180));

  if (direction === 'north') {
    nextDec = Math.min(90, nextDec + stepDeg);
  } else if (direction === 'south') {
    nextDec = Math.max(-90, nextDec - stepDeg);
  } else if (direction === 'east') {
    nextRa = wrapRaDeg(nextRa + stepDeg / cosDec);
  } else if (direction === 'west') {
    nextRa = wrapRaDeg(nextRa - stepDeg / cosDec);
  }

  return quantizeGridCenter(nextRa, nextDec, stepDeg);
}
