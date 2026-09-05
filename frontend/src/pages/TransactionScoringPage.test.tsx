import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DEGRADED_RESULT_FIXTURE,
  INVALID_INPUT_RESULT_FIXTURE,
  MODEL_INFO_FIXTURE,
  OK_RESULT_FIXTURE,
} from '../test/fixtures'
import { renderWithProviders } from '../test/testUtils'
import { TransactionScoringPage } from './TransactionScoringPage'

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return { ...actual, scoreTransaction: vi.fn(), fetchModelInfo: vi.fn() }
})

import { fetchModelInfo, scoreTransaction } from '../api'

async function fillAndSubmit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('radio', { name: /^BALANCED/i }))
  await user.type(screen.getByLabelText('amount'), '1500000')
  await user.type(screen.getByLabelText('oldbalanceOrg'), '1500000')
  await user.type(screen.getByLabelText('newbalanceOrig'), '0')
  await user.type(screen.getByLabelText('oldbalanceDest'), '0')
  await user.type(screen.getByLabelText('newbalanceDest'), '0')
  await user.click(screen.getByRole('button', { name: /score transaction/i }))
}

describe('TransactionScoringPage', () => {
  beforeEach(() => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'success', data: MODEL_INFO_FIXTURE })
  })

  it('renders the operating-mode selector with BALANCED selected by default', () => {
    renderWithProviders(<TransactionScoringPage />)
    const balanced = screen.getByRole('radio', { name: /BALANCED/i })
    expect(balanced).toHaveAttribute('aria-checked', 'true')
  })

  it('happy path: renders risk band, score, action, and top reasons on status="ok"', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ kind: 'success', data: OK_RESULT_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByText(/HIGH RISK/)).toBeInTheDocument())
    expect(screen.getByText('0.895437')).toBeInTheDocument()
    expect(screen.getByText('REVIEW')).toBeInTheDocument()
    for (const reason of OK_RESULT_FIXTURE.top_reasons) {
      expect(screen.getByText(reason)).toBeInTheDocument()
    }
  })

  it('invalid_input: shows the domain-validation message distinctly from a shape error', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ kind: 'success', data: INVALID_INPUT_RESULT_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByText(/rejected: invalid transaction/i)).toBeInTheDocument())
    expect(screen.getByText(INVALID_INPUT_RESULT_FIXTURE.message!)).toBeInTheDocument()
    // must NOT render a risk band for an invalid_input response
    expect(screen.queryByText(/RISK$/)).not.toBeInTheDocument()
  })

  it('degraded: shows a "model temporarily unavailable" state, not a risk score', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ kind: 'success', data: DEGRADED_RESULT_FIXTURE })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByText(/model temporarily unavailable/i)).toBeInTheDocument())
    expect(screen.getByText(/not a low-risk result/i)).toBeInTheDocument()
    expect(screen.queryByText(/RISK$/)).not.toBeInTheDocument()
  })

  it('network error: shows a connectivity message distinct from any API status', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ kind: 'network_error', message: 'Could not reach the API: fetch failed' })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByText(/could not reach the server/i)).toBeInTheDocument())
    expect(screen.getByText(/Could not reach the API: fetch failed/)).toBeInTheDocument()
  })

  it('auth error (401): shows a distinct "authentication failed" state, not a network or degraded one', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({ kind: 'http_error', status: 401, detail: 'Unauthorized' })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() => expect(screen.getByText(/authentication failed/i)).toBeInTheDocument())
    expect(screen.getByText('Unauthorized')).toBeInTheDocument()
    expect(screen.queryByText(/model temporarily unavailable/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/could not reach the server/i)).not.toBeInTheDocument()
  })

  it('shape error: renders per-field messages, not a generic toast', async () => {
    vi.mocked(scoreTransaction).mockResolvedValue({
      kind: 'shape_error',
      errors: [{ type: 'greater_than_equal', loc: ['body', 'amount'], msg: 'Input should be greater than or equal to 0' }],
    })
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)

    await fillAndSubmit(user)

    await waitFor(() =>
      expect(screen.getByText('Input should be greater than or equal to 0')).toBeInTheDocument(),
    )
    // the message appears right under the amount field, not as a page-level generic error
    const amountInput = screen.getByLabelText('amount')
    expect(amountInput).toHaveClass('has-error')
  })

  it('lets the user change operating mode with one click', async () => {
    const user = userEvent.setup()
    renderWithProviders(<TransactionScoringPage />)
    const highPrecision = screen.getByRole('radio', { name: /HIGH PRECISION/i })
    await user.click(highPrecision)
    expect(highPrecision).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: /^BALANCED/i })).toHaveAttribute('aria-checked', 'false')
  })
})
