import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import {
  ROOT_CAUSE_NO_DRIVER_FIXTURE,
  SPIKE_LOOKUP_FIXTURE,
  SPIKE_UNAVAILABLE_FIXTURE,
} from '../test/fixtures'
import { renderWithProviders } from '../test/testUtils'
import { SpikeAnalysisPage } from './SpikeAnalysisPage'

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return { ...actual, analyzeSpike: vi.fn(), fetchModelInfo: vi.fn() }
})

import { analyzeSpike, fetchModelInfo } from '../api'

describe('SpikeAnalysisPage', () => {
  it('happy path: renders the statistical and Isolation Forest signals as SEPARATE chips', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({ kind: 'success', data: SPIKE_LOOKUP_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.type(screen.getByLabelText('step'), '5')
    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() => expect(screen.getByText('Statistical (z-score) signal')).toBeInTheDocument())
    expect(screen.getByText('Isolation Forest signal')).toBeInTheDocument()
    // never collapsed into one combined score element
    expect(screen.queryByText(/combined spike score/i)).not.toBeInTheDocument()
  })

  it('shows spike_context.available=false explicitly rather than hiding the section', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({ kind: 'success', data: SPIKE_UNAVAILABLE_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.type(screen.getByLabelText('step'), '5')
    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() => expect(screen.getByText(/spike detector unavailable/i)).toBeInTheDocument())
  })

  it('shows the explicit "no single driver" negative finding rather than an empty section', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({ kind: 'success', data: ROOT_CAUSE_NO_DRIVER_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.click(screen.getByLabelText(/also run root-cause analysis/i))
    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() =>
      expect(screen.getByText(/no single transaction-type driver exceeded its historical baseline/i)).toBeInTheDocument(),
    )
    expect(screen.getByText('no single driver identified')).toBeInTheDocument()
  })

  it('shape error: shows an invalid-request message when neither field is supplied', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({
      kind: 'shape_error',
      errors: [{ type: 'value_error', loc: ['body'], msg: "either 'step' or 'step_aggregate' must be provided" }],
    })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() => expect(screen.getByText(/invalid request/i)).toBeInTheDocument())
  })

  it('auth error (401): shows a distinct "authentication failed" state', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({ kind: 'http_error', status: 401, detail: 'Unauthorized' })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.type(screen.getByLabelText('step'), '5')
    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() => expect(screen.getByText(/authentication failed/i)).toBeInTheDocument())
    expect(screen.queryByText(/could not reach the server/i)).not.toBeInTheDocument()
  })

  it('network error: shows a connectivity message', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'n/a' })
    vi.mocked(analyzeSpike).mockResolvedValue({ kind: 'network_error', message: 'Could not reach the API' })
    const user = userEvent.setup()
    renderWithProviders(<SpikeAnalysisPage />)

    await user.type(screen.getByLabelText('step'), '5')
    await user.click(screen.getByRole('button', { name: /analyze/i }))

    await waitFor(() => expect(screen.getByText(/could not reach the server/i)).toBeInTheDocument())
  })
})
