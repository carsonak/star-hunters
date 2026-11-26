import React, { useState } from 'react';
import { useNotifications } from '../context/NotificationContext';
import {
  AlertTriangleIcon,
  ChevronIcon,
  CloseIcon,
  HelpQuestionIcon,
  SpinnerIcon,
} from './icons/ObservatoryIcons';

/**
 * Global HUD notification overlay rendering the top-center warning/error banner,
 * bottom-right async operation toast stack, tour prompt toast, and persistent
 * guided-tour launcher button.
 *
 * @returns Rendered notification banners, async toasts, and help FAB.
 */
export const NotificationOverlay: React.FC = () => {
  const {
    alerts,
    dismissWarning,
    asyncTasks,
    finishAsyncTask,
    startTour,
    tourPromptVisible,
    dismissTourPrompt,
  } = useNotifications();

  const [historyExpanded, setHistoryExpanded] = useState(false);

  const latestAlert = alerts[0] ?? null;

  return (
    <>
      {/* Top-Center Warning / Error Notification Banner (Frame 32) */}
      {latestAlert && (
        <div
          className="warning-banner-wrap"
          role="alert"
          aria-live="assertive"
          data-testid="warning-notification-banner"
        >
          <div className={`warning-banner-card ${latestAlert.level === 'error' ? 'is-error' : ''}`}>
            <div className="warning-banner-row">
              <span className="warning-banner-icon">
                <AlertTriangleIcon size={16} />
              </span>

              <div className="warning-banner-body">
                <div className="warning-banner-title">{latestAlert.title}</div>
                {latestAlert.detail && (
                  <div className="warning-banner-detail">{latestAlert.detail}</div>
                )}
                {latestAlert.actions && latestAlert.actions.length > 0 && (
                  <div className="warning-banner-cta-row">
                    {latestAlert.actions.map((action) => (
                      <button
                        key={action.label}
                        type="button"
                        className="hud-btn-secondary warning-banner-cta-btn"
                        onClick={() => {
                          action.onClick();
                          dismissWarning(latestAlert.id);
                        }}
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="warning-banner-actions">
                {alerts.length > 1 && (
                  <button
                    type="button"
                    className="hud-icon-btn"
                    aria-label={`Toggle ${alerts.length} active warnings`}
                    aria-expanded={historyExpanded}
                    data-testid="warning-dropdown-toggle"
                    onClick={() => setHistoryExpanded((prev) => !prev)}
                  >
                    <ChevronIcon size={14} direction={historyExpanded ? 'up' : 'down'} />
                  </button>
                )}

                <button
                  type="button"
                  className="hud-icon-btn"
                  aria-label="Dismiss alert"
                  onClick={() => dismissWarning(latestAlert.id)}
                >
                  <CloseIcon size={14} />
                </button>
              </div>
            </div>

            {historyExpanded && alerts.length > 1 && (
              <div className="warning-history-list">
                {alerts.slice(1).map((item) => (
                  <div key={item.id} className="warning-history-item">
                    <span>
                      <strong className="warning-history-title">{item.title}</strong>
                      {item.detail ? `: ${item.detail}` : ''}
                    </span>
                    <button
                      type="button"
                      className="hud-icon-btn"
                      aria-label={`Dismiss ${item.title}`}
                      onClick={() => dismissWarning(item.id)}
                    >
                      <CloseIcon size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Bottom-Right Async Operations Toast Stack (Frame 33_2) */}
      {(asyncTasks.length > 0 || tourPromptVisible) && (
        <div
          className="async-toast-stack"
          role="status"
          aria-live="polite"
          data-testid="async-toast-stack"
          onPointerDown={(e) => e.stopPropagation()}
          onPointerUp={(e) => e.stopPropagation()}
        >
          {asyncTasks.map((task) => {
            const isSuccess = task.status === 'success';
            return (
              <div key={task.id} className={`async-toast-card ${isSuccess ? 'is-success' : ''}`}>
                {isSuccess ? (
                  <span aria-hidden="true" className="async-toast-success-dot" />
                ) : (
                  <span className="async-toast-spinner">
                    <SpinnerIcon size={14} />
                  </span>
                )}
                <span>{task.label}</span>
                <button
                  type="button"
                  className="hud-icon-btn async-toast-close-btn"
                  aria-label={`Close ${task.label} notification`}
                  onClick={() => finishAsyncTask(task.id)}
                >
                  <CloseIcon size={12} />
                </button>
              </div>
            );
          })}

          {tourPromptVisible && (
            <div className="async-toast-card is-tour-prompt" data-testid="tour-prompt-toast">
              <span>New to Star Hunters? Take the 60s guided observatory tour.</span>
              <button
                type="button"
                className="hud-btn-primary async-toast-tour-btn"
                onClick={() => startTour(0)}
              >
                Start Tour
              </button>
              <button
                type="button"
                className="hud-icon-btn async-toast-close-btn"
                aria-label="Dismiss tour prompt"
                onClick={dismissTourPrompt}
              >
                <CloseIcon size={12} />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Persistent 40x40px Bottom-Right Guided Tour Launcher (image 6) */}
      <button
        type="button"
        className="help-fab-btn"
        aria-label="Open Guided Observatory Tour"
        title="Launch Guided Tour"
        data-testid="guided-tour-help-btn"
        onPointerDown={(e) => e.stopPropagation()}
        onPointerUp={(e) => e.stopPropagation()}
        onClick={() => startTour(0)}
      >
        <HelpQuestionIcon size={19} />
      </button>
    </>
  );
};
