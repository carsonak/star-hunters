import React, { useMemo } from 'react';
import type { ComparisonMode, MissionName, ViewportRenderSettings } from '../../types/contracts';
import {
  isWavelengthPresetActive,
  isWavelengthPresetEnabled,
  resolveWavelengthConfigForMissions,
  type MissionSourceCount,
} from '../../utils/knownObjectSelection';
import { DualThumbRangeSlider } from '../DualThumbRangeSlider';
import { ChevronIcon } from '../icons/ObservatoryIcons';

/** Props for {@link BlinkerControlsSection}. */
export interface BlinkerControlsSectionProps {
  /** Whether the controls pane is collapsed. */
  controlsFolded: boolean;
  /** Whether the sibling known-objects pane is collapsed (controls flex sizing). */
  objectsFolded: boolean;
  /** Normalized `[0.25, 0.75]` height ratio allocated to this pane. */
  splitRatio: number;
  /** Toggle callback for the section header accordion button. */
  onToggleControlsFold: () => void;
  /** Current Blinker render and comparison settings. */
  renderSettings: ViewportRenderSettings;
  /** Partial updater for viewport render settings. */
  updateRenderSettings: (partial: Partial<ViewportRenderSettings>) => void;
  /** Epoch indices permitted by the active mission filter. */
  allowedEpochIndices: readonly number[];
  /** Distinct missions and epoch counts present in the loaded patch. */
  availableMissions: ReadonlyArray<MissionSourceCount>;
  /** Currently enabled mission filters. */
  activeMissions: ReadonlyArray<MissionName>;
  /** Primary mission reported by the patch manifest (`data_source_summary.primary_mission`). */
  primaryMission?: MissionName;
  /** Optional handler to toggle a mission filter pill. */
  toggleMissionFilter?: (mission: MissionName) => void;
}

const COMPARISON_MODES: ReadonlyArray<{ id: ComparisonMode; label: string }> = [
  { id: 'blink', label: 'Blink' },
  { id: 'subtract', label: 'Subtract' },
  { id: 'coadd', label: 'Add' },
];

/**
 * Collapsible top pane of {@link BlinkerSidebar} containing the Dim & Pop
 * wavelength slice slider, quick presets, comparison mode selector, pass pair
 * span slider, and mission source filter pills.
 *
 * @param props - Accordion state, render settings, epoch/mission lists, and handlers.
 * @returns Rendered Pass & Band Controls section.
 */
export const BlinkerControlsSection: React.FC<BlinkerControlsSectionProps> = ({
  controlsFolded,
  objectsFolded,
  splitRatio,
  onToggleControlsFold,
  renderSettings,
  updateRenderSettings,
  allowedEpochIndices,
  availableMissions,
  activeMissions,
  primaryMission,
  toggleMissionFilter,
}) => {
  const onlyOneMissionActive = useMemo(
    () => availableMissions.filter((m) => activeMissions.includes(m.mission)).length <= 1,
    [activeMissions, availableMissions],
  );
  const wavelengthConfig = useMemo(
    () => resolveWavelengthConfigForMissions(availableMissions, activeMissions, primaryMission),
    [activeMissions, availableMissions, primaryMission],
  );

  const passPosA = Math.max(0, allowedEpochIndices.indexOf(renderSettings.passAIndex));
  const passPosB = Math.max(passPosA + 1, allowedEpochIndices.indexOf(renderSettings.passBIndex));

  return (
    <section
      className={`blinker-section-pane ${
        controlsFolded ? 'is-folded' : objectsFolded ? 'is-expanded' : ''
      }`}
      style={!controlsFolded && !objectsFolded ? { flex: `${splitRatio} 1 0%` } : undefined}
    >
      <button
        type="button"
        className="blinker-section-header"
        aria-expanded={!controlsFolded}
        data-testid="blinker-controls-chevron"
        onClick={onToggleControlsFold}
      >
        <ChevronIcon size={13} direction={controlsFolded ? 'right' : 'down'} />
        <span>Pass &amp; Band Controls</span>
      </button>

      {!controlsFolded && (
        <div className="blinker-section-body">
          <div>
            <DualThumbRangeSlider
              label="Wavelength Slice"
              variant="spectrum"
              min={wavelengthConfig.railMin}
              max={wavelengthConfig.railMax}
              allowedMin={wavelengthConfig.min}
              allowedMax={wavelengthConfig.max}
              step={wavelengthConfig.step}
              minGap={wavelengthConfig.minGap}
              value={renderSettings.wavelengthRangeUm}
              formatReadout={(lo, hi) => `${lo.toFixed(2)}-${hi.toFixed(2)} µm`}
              lowAriaLabel="Minimum wavelength micrometers"
              highAriaLabel="Maximum wavelength micrometers"
              testId="wavelength-dual-slider"
              onChange={(next) => updateRenderSettings({ wavelengthRangeUm: next })}
            />
            <div className="blinker-preset-row">
              {wavelengthConfig.presets.map((preset) => {
                const isEnabled = isWavelengthPresetEnabled(preset.range, wavelengthConfig);
                const isActive =
                  isEnabled &&
                  isWavelengthPresetActive(renderSettings.wavelengthRangeUm, preset.range);
                return (
                  <button
                    key={preset.id}
                    type="button"
                    disabled={!isEnabled}
                    title={
                      isEnabled
                        ? preset.label
                        : `${preset.label} — outside active mission range (${wavelengthConfig.min.toFixed(2)}–${wavelengthConfig.max.toFixed(2)} µm)`
                    }
                    aria-pressed={isActive}
                    className={`hud-btn-secondary blinker-preset-btn ${isActive ? 'is-active' : ''}`}
                    onClick={() => updateRenderSettings({ wavelengthRangeUm: preset.range })}
                  >
                    {preset.shortLabel}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="blinker-mode-stack">
            <div
              className="mode-segmented-pill"
              role="tablist"
              aria-label="Comparison Mode"
              data-testid="comparison-mode-pill"
            >
              {COMPARISON_MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="tab"
                  aria-selected={renderSettings.mode === m.id}
                  data-testid={`mode-btn-${m.id}`}
                  className={`mode-segmented-btn ${
                    renderSettings.mode === m.id ? 'is-active' : ''
                  }`}
                  onClick={() => updateRenderSettings({ mode: m.id })}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {allowedEpochIndices.length >= 2 && (
              <DualThumbRangeSlider
                label={renderSettings.mode === 'coadd' ? 'Co-Add Pass Span' : 'Pass Pair Window'}
                variant="plain"
                min={1}
                max={allowedEpochIndices.length}
                step={1}
                minGap={1}
                value={[
                  Math.min(passPosA + 1, allowedEpochIndices.length - 1),
                  Math.max(2, Math.min(allowedEpochIndices.length, passPosB + 1)),
                ]}
                formatReadout={(lo, hi) => {
                  const idxA = allowedEpochIndices[lo - 1] ?? 0;
                  const idxB =
                    allowedEpochIndices[hi - 1] ??
                    allowedEpochIndices[allowedEpochIndices.length - 1] ??
                    1;
                  return renderSettings.mode === 'subtract'
                    ? `P${idxB + 1} - P${idxA + 1}`
                    : `P${idxA + 1} .. P${idxB + 1}`;
                }}
                lowAriaLabel="Left reference pass"
                highAriaLabel="Right target pass"
                testId="pass-dual-slider"
                onChange={([lo, hi]) => {
                  const posA = Math.max(0, Math.min(allowedEpochIndices.length - 2, lo - 1));
                  const posB = Math.max(posA + 1, Math.min(allowedEpochIndices.length - 1, hi - 1));
                  const idxA = allowedEpochIndices[posA] ?? 0;
                  const idxB = allowedEpochIndices[posB] ?? idxA;
                  updateRenderSettings({
                    passAIndex: idxA,
                    passBIndex: idxB,
                    coaddIndices: allowedEpochIndices.slice(posA, posB + 1),
                  });
                }}
              />
            )}
          </div>

          <div>
            <div className="control-section-header control-section-header--compact">
              <span className="control-section-title">Mission Sources</span>
            </div>
            <div className="mission-filter-row" data-testid="mission-filter-row">
              {availableMissions.map(({ mission, count }) => {
                const enabled = activeMissions.includes(mission);
                const isLockedLast = enabled && onlyOneMissionActive;
                return (
                  <button
                    key={mission}
                    type="button"
                    aria-pressed={enabled}
                    data-testid={`mission-pill-${mission}`}
                    className={`mission-pill-btn ${enabled ? 'is-enabled' : ''}`}
                    disabled={isLockedLast}
                    title={
                      isLockedLast
                        ? 'At least one active mission source must remain enabled'
                        : `${enabled ? 'Hide' : 'Show'} ${mission} (${count} passes)`
                    }
                    onClick={() => toggleMissionFilter?.(mission)}
                  >
                    <span className="mission-pill-dot" />
                    <span>
                      {mission} ({count})
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
