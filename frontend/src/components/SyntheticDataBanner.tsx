import { useModelInfo } from '../context/ModelInfoContext'

export function SyntheticDataBanner() {
  const { syntheticDataCaveat } = useModelInfo()
  return (
    <div className="caveat-banner" role="note" aria-label="Synthetic data disclosure">
      <strong>Prototype notice:</strong> {syntheticDataCaveat}
    </div>
  )
}
