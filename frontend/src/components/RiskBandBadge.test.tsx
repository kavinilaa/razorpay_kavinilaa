import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { RiskBandBadge } from './RiskBandBadge'

describe('RiskBandBadge', () => {
  it('renders a text label in addition to color, for every band', () => {
    const { rerender } = render(<RiskBandBadge band="LOW" />)
    expect(screen.getByText(/LOW RISK/)).toBeInTheDocument()

    rerender(<RiskBandBadge band="MEDIUM" />)
    expect(screen.getByText(/MEDIUM RISK/)).toBeInTheDocument()

    rerender(<RiskBandBadge band="HIGH" />)
    expect(screen.getByText(/HIGH RISK/)).toBeInTheDocument()
  })

  it('uses a different CSS class per band (not color alone)', () => {
    const { container, rerender } = render(<RiskBandBadge band="LOW" />)
    expect(container.querySelector('.risk-LOW')).toBeInTheDocument()

    rerender(<RiskBandBadge band="HIGH" />)
    expect(container.querySelector('.risk-HIGH')).toBeInTheDocument()
    expect(container.querySelector('.risk-LOW')).not.toBeInTheDocument()
  })
})
