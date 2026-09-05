import { useModelInfo } from '../context/ModelInfoContext'

/**
 * Persistent, always-visible reminder that this is a prototype on synthetic
 * data - rendered in the app header on EVERY page (see Layout.tsx), not
 * confined to the Model Info view. Never goes blank: falls back to a
 * hard-coded caveat if GET /model-info hasn't loaded yet or failed.
 */
export function SyntheticDataBanner() {
  const { syntheticDataCaveat } = useModelInfo()
  return (
    <div className="caveat-banner" role="note" aria-label="Synthetic data disclosure">
      <strong>Prototype notice:</strong> {syntheticDataCaveat}
    </div>
  )
}
