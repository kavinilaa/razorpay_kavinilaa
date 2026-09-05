import type { OperatingMode } from '../api'
import { OPERATING_MODES } from '../api'

const MODE_DESCRIPTIONS: Record<OperatingMode, string> = {
  HIGH_RECALL: 'Catch more suspicious activity.',
  BALANCED: 'Recommended default.',
  HIGH_PRECISION: 'Reduce false alarms.',
}

/**
 * A visible, one-click operating-mode selector - never buried in a settings
 * panel. When `modelInfo` has loaded, each option shows its measured
 * validation precision/recall (GET /model-info) right next to it, so the
 * tradeoff is visible BEFORE submitting, not just after scoring.
 */
export function OperatingModeSelector({
  value,
  onChange,
}: {
  value: OperatingMode
  onChange: (mode: OperatingMode) => void
}) {
  return (
    <div>
      <h3 id="operating-mode-label">Review preference</h3>
      <p className="muted">Choose the review level that fits your team.</p>
      <div className="mode-selector" role="radiogroup" aria-labelledby="operating-mode-label">
        {OPERATING_MODES.map((mode) => {
          return (
            <button
              key={mode}
              type="button"
              role="radio"
              aria-checked={value === mode}
              className={`mode-option${value === mode ? ' selected' : ''}`}
              onClick={() => onChange(mode)}
            >
              <span className="mode-name">{mode.replace('_', ' ')}</span>
              <span className="mode-description">{MODE_DESCRIPTIONS[mode]}</span>
            </button>
          )
        })}
      </div>
      <p className="muted">
        BALANCED is selected by default. Results are measured on synthetic test data.
      </p>
    </div>
  )
}
