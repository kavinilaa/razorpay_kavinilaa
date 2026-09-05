import { screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { HEALTHY_FIXTURE, MODEL_INFO_FIXTURE } from '../test/fixtures'
import { renderWithProviders } from '../test/testUtils'
import { ModelInfoPage } from './ModelInfoPage'

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return { ...actual, fetchModelInfo: vi.fn(), fetchHealth: vi.fn() }
})

import { fetchHealth, fetchModelInfo } from '../api'

describe('ModelInfoPage', () => {
  it('happy path: renders data_provenance, the synthetic-data caveat, and a healthy status', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'success', data: MODEL_INFO_FIXTURE })
    vi.mocked(fetchHealth).mockResolvedValue({ kind: 'success', data: HEALTHY_FIXTURE })

    renderWithProviders(<ModelInfoPage />)

    await waitFor(() => expect(screen.getByText(MODEL_INFO_FIXTURE.data_provenance)).toBeInTheDocument())
    expect(screen.getByText(MODEL_INFO_FIXTURE.synthetic_data_caveat)).toBeInTheDocument()
    expect(screen.getByText('HEALTHY')).toBeInTheDocument()
    expect(screen.getByText('transaction_model')).toBeInTheDocument()
  })

  it('shows which component failed when health is degraded', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'success', data: MODEL_INFO_FIXTURE })
    vi.mocked(fetchHealth).mockResolvedValue({
      kind: 'success',
      data: {
        ...HEALTHY_FIXTURE,
        status: 'degraded',
        components: {
          ...HEALTHY_FIXTURE.components,
          transaction_model: { loaded: false, path: 'models/xgboost.joblib', error: 'model file not found' },
        },
      },
    })

    renderWithProviders(<ModelInfoPage />)

    await waitFor(() => expect(screen.getByText('DEGRADED')).toBeInTheDocument())
    expect(screen.getByText('model file not found')).toBeInTheDocument()
  })

  it('model-info network error: shows a connectivity message with a retry option', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'Could not reach the API' })
    vi.mocked(fetchHealth).mockResolvedValue({ kind: 'success', data: HEALTHY_FIXTURE })

    renderWithProviders(<ModelInfoPage />)

    await waitFor(() => expect(screen.getAllByText(/could not reach the server/i).length).toBeGreaterThan(0))
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
  })
})
