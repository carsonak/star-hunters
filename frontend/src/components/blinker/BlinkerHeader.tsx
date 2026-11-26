import React from 'react';
import type {
  BinaryDtype,
  ContrastStretch,
  FetchQualityMode,
  ImageProvenanceBadge,
} from '../../types/contracts';
import {
  ChevronIcon,
  CloseIcon,
  ResetViewIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from '../icons/ObservatoryIcons';

interface BlinkerHeaderProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  zoom: number;
  panX: number;
  panY: number;
  contrastStretch: ContrastStretch;
  fetchQualityMode: FetchQualityMode;
  activePatchDtype: BinaryDtype;
  isNetworkSlow: boolean;
  isRefetching: boolean;
  title: string;
  fullSubtitleText: string;
  provenance: ImageProvenanceBadge;
  statusNote?: string;
  onSetZoomPan: (zoom: number, panX: number, panY: number) => void;
  onResetZoomPan: () => void;
  onChangeContrast: (next: ContrastStretch) => void;
  onChangeQualityMode: (next: FetchQualityMode) => void;
  onRefetchPatch: () => void;
  onClose?: () => void;
}

/** Modal header containing sidebar toggle, viewport zoom/stretch controls, marquee subtitle, quality/refetch controls, and provenance pill. */
export const BlinkerHeader: React.FC<BlinkerHeaderProps> = ({
  sidebarCollapsed,
  onToggleSidebar,
  zoom,
  panX,
  panY,
  contrastStretch,
  fetchQualityMode,
  activePatchDtype,
  isNetworkSlow,
  isRefetching,
  title,
  fullSubtitleText,
  provenance,
  statusNote,
  onSetZoomPan,
  onResetZoomPan,
  onChangeContrast,
  onChangeQualityMode,
  onRefetchPatch,
  onClose,
}) => (
  <header
    className="blinker-header"
    onPointerDown={(e) => e.stopPropagation()}
    onPointerUp={(e) => e.stopPropagation()}
  >
    <div className="blinker-header-left">
      <button
        type="button"
        className={`hud-icon-btn ${!sidebarCollapsed ? 'is-active' : ''}`}
        aria-label={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
        aria-expanded={!sidebarCollapsed}
        data-testid="blinker-sidebar-toggle"
        title={sidebarCollapsed ? 'Show Controls & Known Objects Sidebar' : 'Hide Sidebar'}
        onClick={onToggleSidebar}
      >
        <ChevronIcon size={14} direction={sidebarCollapsed ? 'right' : 'left'} />
      </button>
      <button
        type="button"
        className="hud-icon-btn"
        aria-label="Zoom out cutout (- or _)"
        title="Zoom out (- / _)"
        onClick={() => onSetZoomPan(zoom / 1.4, panX, panY)}
      >
        <ZoomOutIcon size={14} />
      </button>
      <span className="mono" style={{ minWidth: 34, textAlign: 'center' }}>
        {zoom.toFixed(1)}×
      </span>
      <button
        type="button"
        className="hud-icon-btn"
        aria-label="Zoom in cutout (+ or =)"
        title="Zoom in (+ / =)"
        onClick={() => onSetZoomPan(zoom * 1.4, panX, panY)}
      >
        <ZoomInIcon size={14} />
      </button>
      {zoom > 1.01 && (
        <button
          type="button"
          className="hud-icon-btn"
          aria-label="Reset zoom and pan (0)"
          title="Reset zoom and pan (0)"
          onClick={onResetZoomPan}
        >
          <ResetViewIcon size={14} />
        </button>
      )}
      <select
        aria-label="Contrast stretch"
        className="hud-select"
        style={{ height: 26, fontSize: 11, marginLeft: 4 }}
        value={contrastStretch}
        onChange={(e) => onChangeContrast(e.target.value as ContrastStretch)}
      >
        <option value="asinh">Asinh</option>
        <option value="zscale">ZScale</option>
        <option value="linear">Linear</option>
      </select>
    </div>

    <div className="blinker-section-label" title={fullSubtitleText}>
      <span className="blinker-section-title">{title}</span>
      <div className="blinker-section-sub-marquee" tabIndex={0}>
        <span className="blinker-section-sub mono">{fullSubtitleText}</span>
      </div>
    </div>

    <div className="blinker-header-right">
      <select
        aria-label="Patch Fetch Quality"
        data-testid="blinker-quality-select"
        className="hud-select"
        style={{ height: 26, fontSize: 11 }}
        title="Frame download quality (Auto switches to Compressed below 3 Mbps)"
        value={fetchQualityMode}
        onChange={(e) => onChangeQualityMode(e.target.value as FetchQualityMode)}
      >
        <option value="auto">Auto</option>
        <option value="full">Full</option>
        <option value="compressed">Compressed</option>
      </select>
      <span
        className={`blinker-dtype-pill ${
          activePatchDtype === 'float16' ? 'is-degraded' : 'is-full'
        }`}
        data-testid="blinker-dtype-badge"
        title={
          activePatchDtype === 'float16'
            ? 'Loaded at compressed precision (reduced dynamic range)'
            : 'Loaded at full scientific precision'
        }
      >
        {activePatchDtype === 'float16'
          ? isNetworkSlow
            ? 'Compressed · Low BW'
            : 'Compressed'
          : 'Full'}
      </span>
      <button
        type="button"
        className="hud-btn-secondary"
        data-testid="blinker-refetch-btn"
        disabled={isRefetching}
        title="Refetch this sky patch bypassing local browser cache"
        style={{ height: 26, padding: '0 8px', fontSize: 11 }}
        onClick={onRefetchPatch}
      >
        {isRefetching ? 'Refetching…' : 'Refetch'}
      </button>
      <span
        className={`provenance-pill provenance-pill--${provenance.tier}`}
        data-testid="blinker-provenance-badge"
        title={statusNote ? `${provenance.label} (${statusNote})` : provenance.label}
      >
        {provenance.label}
      </span>
      <button
        type="button"
        className="hud-icon-btn"
        aria-label="Close Blinker Modal"
        data-testid="blinker-close-btn"
        onClick={onClose}
      >
        <CloseIcon size={16} />
      </button>
    </div>
  </header>
);
