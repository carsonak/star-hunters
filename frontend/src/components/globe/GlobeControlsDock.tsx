import React from 'react';
import type { FetchQualityMode } from '../../types/contracts';
import { SUPPORTED_GLOBE_PROJECTIONS, type GlobeProjection } from '../../utils/globeNavigation';

interface GlobeControlsDockProps {
  showConstellations: boolean;
  projection: GlobeProjection;
  fetchQualityMode: FetchQualityMode;
  onPan: (dir: 'north' | 'south' | 'east' | 'west') => void;
  onZoom: (factor: number | 'reset') => void;
  onToggleConstellations: () => void;
  onChangeProjection: (nextProj: GlobeProjection) => void;
  onChangeQualityMode: (nextMode: FetchQualityMode) => void;
  onOpenCenterPatch: () => void;
}

/** Accessible pan/zoom cross, constellation toggle, projection, quality, and center-patch launcher. */
export const GlobeControlsDock: React.FC<GlobeControlsDockProps> = ({
  showConstellations,
  projection,
  fetchQualityMode,
  onPan,
  onZoom,
  onToggleConstellations,
  onChangeProjection,
  onChangeQualityMode,
  onOpenCenterPatch,
}) => (
  <div
    className="globe-controls-dock"
    data-testid="globe-nav-controls"
    role="group"
    aria-label="Sky Globe Pan, Zoom and Overlay Controls"
    onPointerDown={(e) => e.stopPropagation()}
    onPointerUp={(e) => e.stopPropagation()}
    onClick={(e) => e.stopPropagation()}
  >
    <div className="globe-pan-cross" aria-label="Pan Sky Coordinates (North-Up)">
      <span />
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Pan Up / North (Arrow Up or W)"
        title="Pan Up / North (↑ or W; Shift for fine step)"
        data-testid="globe-pan-north"
        onClick={() => onPan('north')}
      >
        ↑
      </button>
      <span />
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Pan Left / East (Arrow Left or A)"
        title="Pan Left / East (← or A; Shift for fine step)"
        data-testid="globe-pan-east"
        onClick={() => onPan('east')}
      >
        ←
      </button>
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Pan Down / South (Arrow Down or S)"
        title="Pan Down / South (↓ or S; Shift for fine step)"
        data-testid="globe-pan-south"
        onClick={() => onPan('south')}
      >
        ↓
      </button>
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Pan Right / West (Arrow Right or D)"
        title="Pan Right / West (→ or D; Shift for fine step)"
        data-testid="globe-pan-west"
        onClick={() => onPan('west')}
      >
        →
      </button>
    </div>

    <span className="globe-telemetry-divider" style={{ height: 24 }} />

    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Zoom In (+ or =)"
        title="Zoom In (+ / =)"
        data-testid="globe-zoom-in"
        onClick={() => onZoom(0.62)}
      >
        +
      </button>
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Zoom Out (- or _)"
        title="Zoom Out (- / _)"
        data-testid="globe-zoom-out"
        onClick={() => onZoom(1.48)}
      >
        -
      </button>
      <button
        type="button"
        className="globe-nav-btn"
        aria-label="Reset Sky Zoom (0)"
        title="Reset View — 70° FoV (0)"
        data-testid="globe-zoom-reset"
        onClick={() => onZoom('reset')}
      >
        Reset
      </button>
    </div>

    <span className="globe-telemetry-divider" style={{ height: 24 }} />

    <button
      type="button"
      className={`globe-nav-btn ${showConstellations ? 'is-active' : ''}`}
      aria-pressed={showConstellations}
      aria-label="Toggle Constellation Lines and Labels"
      data-testid="globe-constellations-toggle"
      onClick={onToggleConstellations}
    >
      Constellations
    </button>

    <select
      aria-label="Sky Globe Projection"
      data-testid="globe-projection-select"
      className="globe-nav-btn"
      style={{ paddingRight: 6 }}
      value={projection}
      onChange={(e) => onChangeProjection(e.target.value as GlobeProjection)}
    >
      {SUPPORTED_GLOBE_PROJECTIONS.map((opt) => (
        <option key={opt.id} value={opt.id}>
          {opt.label}
        </option>
      ))}
    </select>

    <span className="globe-telemetry-divider" style={{ height: 24 }} />

    <select
      aria-label="Patch Fetch Quality"
      data-testid="globe-quality-select"
      className="globe-nav-btn"
      style={{ paddingRight: 6 }}
      title="Frame download quality (Auto switches to Compressed below 3 Mbps)"
      value={fetchQualityMode}
      onChange={(e) => onChangeQualityMode(e.target.value as FetchQualityMode)}
    >
      <option value="auto">Quality: Auto</option>
      <option value="full">Quality: Full</option>
      <option value="compressed">Quality: Compressed</option>
    </select>

    <button
      type="button"
      className="hud-btn-primary"
      data-testid="globe-open-center-patch"
      title="Open Blinker detailed view for the sky patch at the center of the viewport"
      style={{ height: 28, padding: '0 10px', fontSize: 11.5 }}
      onClick={onOpenCenterPatch}
    >
      Open Detailed View
    </button>
  </div>
);
