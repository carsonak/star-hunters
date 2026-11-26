import React from 'react';
import type { RecentPatchEntry } from '../../types/contracts';
import { ChevronIcon } from '../icons/ObservatoryIcons';

export interface BlinkerRecentPatchesSectionProps {
  recentsFolded: boolean;
  onToggleRecentsFold: () => void;
  recentPatches: ReadonlyArray<RecentPatchEntry>;
  activePatchId: string | null;
  onSelectRecentPatch: (entry: RecentPatchEntry) => void;
  onClearRecentPatches?: () => void;
}

/**
 * Collapsible VS Code-style Explorer pane rendered at the top of {@link BlinkerSidebar}
 * above Pass & Band Controls, listing recently opened sky patches.
 */
export const BlinkerRecentPatchesSection: React.FC<BlinkerRecentPatchesSectionProps> = ({
  recentsFolded,
  onToggleRecentsFold,
  recentPatches,
  activePatchId,
  onSelectRecentPatch,
  onClearRecentPatches,
}) => (
  <section
    className={`blinker-section-pane ${recentsFolded ? 'is-folded' : 'is-recents-expanded'}`}
    data-testid="blinker-recents-section"
  >
    <button
      type="button"
      className="blinker-section-header"
      aria-expanded={!recentsFolded}
      data-testid="blinker-recents-chevron"
      onClick={onToggleRecentsFold}
    >
      <ChevronIcon size={13} direction={recentsFolded ? 'right' : 'down'} />
      <span>Recent Patches ({recentPatches.length})</span>
    </button>

    {!recentsFolded && (
      <div className="blinker-section-body blinker-section-body--recents">
        {recentPatches.length === 0 ? (
          <div className="blinker-objects-empty">No recently opened patches yet.</div>
        ) : (
          <>
            <div className="blinker-recents-list" data-testid="blinker-recents-list">
              {recentPatches.map((entry) => {
                const isActive = entry.id === activePatchId;
                return (
                  <button
                    key={entry.id}
                    type="button"
                    className={`blinker-recent-row ${isActive ? 'is-active' : ''}`}
                    data-testid={`blinker-recent-item-${entry.id}`}
                    onClick={() => onSelectRecentPatch(entry)}
                    title={`Open ${entry.title} (${entry.coordsLabel})`}
                  >
                    <span className="blinker-recent-title">{entry.title}</span>
                    <span className="mono blinker-recent-meta">
                      {entry.coordsLabel}
                      {entry.mission ? ` · ${entry.mission}` : ''}
                    </span>
                  </button>
                );
              })}
            </div>
            {onClearRecentPatches && (
              <button
                type="button"
                className="hud-btn-secondary blinker-objects-action-btn"
                data-testid="blinker-recents-clear"
                onClick={onClearRecentPatches}
              >
                Clear Recents
              </button>
            )}
          </>
        )}
      </div>
    )}
  </section>
);
