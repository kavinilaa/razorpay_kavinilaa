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
