/**
 * Renders the risk band exactly as the API returned it - no recomputation
 * from risk_score. Color is never the only signal: a text label and a
 * distinct glyph (●▲■) are always present too, so the badge remains legible
 * without color (accessibility).
 */
const GLYPH: Record<string, string> = {
  LOW: '●',
  MEDIUM: '▲',
  HIGH: '■',
}

export function RiskBandBadge({ band }: { band: 'LOW' | 'MEDIUM' | 'HIGH' }) {
  return (
    <span className={`risk-badge risk-${band}`}>
      <span className="glyph" aria-hidden="true">
        {GLYPH[band]}
      </span>
      {band} RISK
    </span>
  )
}
