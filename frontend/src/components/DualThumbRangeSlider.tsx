import React, { useCallback, useRef } from 'react';

export interface DualThumbRangeSliderProps {
  label: string;
  min: number;
  max: number;
  /** Optional inner lower bound for thumb movement on the full `[min, max]` rail. */
  allowedMin?: number;
  /** Optional inner upper bound for thumb movement on the full `[min, max]` rail. */
  allowedMax?: number;
  step: number;
  minGap?: number;
  value: [number, number];
  onChange: (next: [number, number]) => void;
  formatReadout: (low: number, high: number) => string;
  lowAriaLabel?: string;
  highAriaLabel?: string;
  variant?: 'spectrum' | 'plain';
  testId?: string;
}

function quantize(val: number, min: number, max: number, step: number): number {
  const clamped = Math.max(min, Math.min(max, val));
  const steps = Math.round((clamped - min) / step);
  const rounded = min + steps * step;
  return Number(Math.max(min, Math.min(max, rounded)).toFixed(6));
}

export const DualThumbRangeSlider: React.FC<DualThumbRangeSliderProps> = ({
  label,
  min,
  max,
  allowedMin,
  allowedMax,
  step,
  minGap = step,
  value,
  onChange,
  formatReadout,
  lowAriaLabel,
  highAriaLabel,
  variant = 'plain',
  testId,
}) => {
  const railRef = useRef<HTMLDivElement | null>(null);
  const activeThumbRef = useRef<'low' | 'high' | null>(null);

  const effectiveMin = Math.max(min, Math.min(max, allowedMin ?? min));
  const effectiveMax = Math.max(effectiveMin, Math.min(max, allowedMax ?? max));

  const low = Math.max(effectiveMin, Math.min(effectiveMax, value[0]));
  const high = Math.max(effectiveMin, Math.min(effectiveMax, value[1]));
  const span = Math.max(1e-6, max - min);
  const lowPct = ((low - min) / span) * 100;
  const highPct = ((high - min) / span) * 100;

  const valueFromClientX = useCallback(
    (clientX: number): number => {
      const rect = railRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) return low;
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return quantize(min + ratio * (max - min), effectiveMin, effectiveMax, step);
    },
    [effectiveMax, effectiveMin, low, max, min, step],
  );

  const commitThumbMove = useCallback(
    (which: 'low' | 'high', rawTarget: number) => {
      if (which === 'low') {
        const clampedLow = Math.min(rawTarget, high - minGap);
        onChange([quantize(clampedLow, effectiveMin, effectiveMax, step), high]);
      } else {
        const clampedHigh = Math.max(rawTarget, low + minGap);
        onChange([low, quantize(clampedHigh, effectiveMin, effectiveMax, step)]);
      }
    },
    [effectiveMax, effectiveMin, high, low, minGap, onChange, step],
  );

  const handleTrackPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const raw = valueFromClientX(e.clientX);
    const dLow = Math.abs(raw - low);
    const dHigh = Math.abs(raw - high);
    const which: 'low' | 'high' = dLow <= dHigh ? 'low' : 'high';
    activeThumbRef.current = which;
    commitThumbMove(which, raw);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handleTrackPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const which = activeThumbRef.current;
    if (!which) return;
    commitThumbMove(which, valueFromClientX(e.clientX));
  };

  const handleTrackPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!activeThumbRef.current) return;
    activeThumbRef.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore already released capture
    }
  };

  const handleKeyDown = (which: 'low' | 'high') => (e: React.KeyboardEvent<HTMLButtonElement>) => {
    let delta = 0;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') delta = step;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') delta = -step;
    if (delta === 0) return;
    e.preventDefault();
    commitThumbMove(which, (which === 'low' ? low : high) + delta);
  };

  return (
    <div className="dual-slider-root" data-testid={testId}>
      <div className="control-section-header">
        <span className="control-section-title">{label}</span>
        <span className="control-value-readout mono">{formatReadout(low, high)}</span>
      </div>

      <div
        className="dual-slider-track"
        onPointerDown={handleTrackPointerDown}
        onPointerMove={handleTrackPointerMove}
        onPointerUp={handleTrackPointerUp}
        onPointerCancel={handleTrackPointerUp}
      >
        <div
          ref={railRef}
          className={`dual-slider-rail ${
            variant === 'spectrum' ? 'dual-slider-rail--spectrum' : ''
          }`}
        >
          {variant === 'spectrum' && lowPct > 0 && (
            <div className="dual-slider-dim-mask" style={{ left: 0, width: `${lowPct}%` }} />
          )}
          {variant === 'spectrum' && highPct < 100 && (
            <div
              className="dual-slider-dim-mask"
              style={{ left: `${highPct}%`, width: `${Math.max(0, 100 - highPct)}%` }}
            />
          )}
          <div
            className={`dual-slider-active-range ${
              variant === 'spectrum' ? 'is-spectrum' : 'is-plain'
            }`}
            style={{
              left: `${lowPct}%`,
              width: `${Math.max(2, highPct - lowPct)}%`,
            }}
          />
        </div>

        <button
          type="button"
          role="slider"
          className="dual-slider-thumb"
          style={{ left: `${lowPct}%` }}
          aria-label={lowAriaLabel ?? `${label} minimum`}
          aria-valuemin={effectiveMin}
          aria-valuemax={effectiveMax - minGap}
          aria-valuenow={low}
          onPointerDown={(e) => {
            e.stopPropagation();
            activeThumbRef.current = 'low';
            e.currentTarget.parentElement?.setPointerCapture(e.pointerId);
          }}
          onKeyDown={handleKeyDown('low')}
          data-testid={testId ? `${testId}-low` : undefined}
        />
        <button
          type="button"
          role="slider"
          className="dual-slider-thumb"
          style={{ left: `${highPct}%` }}
          aria-label={highAriaLabel ?? `${label} maximum`}
          aria-valuemin={effectiveMin + minGap}
          aria-valuemax={effectiveMax}
          aria-valuenow={high}
          onPointerDown={(e) => {
            e.stopPropagation();
            activeThumbRef.current = 'high';
            e.currentTarget.parentElement?.setPointerCapture(e.pointerId);
          }}
          onKeyDown={handleKeyDown('high')}
          data-testid={testId ? `${testId}-high` : undefined}
        />
      </div>
    </div>
  );
};
