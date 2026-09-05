import { screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { MODEL_INFO_FIXTURE } from '../test/fixtures'
import { renderWithProviders } from '../test/testUtils'
import { SyntheticDataBanner } from './SyntheticDataBanner'

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return { ...actual, fetchModelInfo: vi.fn() }
})

import { fetchModelInfo } from '../api'

describe('SyntheticDataBanner', () => {
  it('shows a non-empty fallback caveat before the model-info fetch resolves', () => {
    vi.mocked(fetchModelInfo).mockReturnValue(new Promise(() => {})) // never resolves
    renderWithProviders(<SyntheticDataBanner />)
    expect(screen.getByRole('note', { name: /synthetic data disclosure/i })).toHaveTextContent(/synthetic/i)
  })

  it('shows the real data_provenance text once model-info resolves', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'success', data: MODEL_INFO_FIXTURE })
    renderWithProviders(<SyntheticDataBanner />)
    await waitFor(() =>
      expect(screen.getByRole('note')).toHaveTextContent(MODEL_INFO_FIXTURE.data_provenance),
    )
  })

  it('falls back gracefully on a network error - the banner is never blank', async () => {
    vi.mocked(fetchModelInfo).mockResolvedValue({ kind: 'network_error', message: 'offline' })
    renderWithProviders(<SyntheticDataBanner />)
    await waitFor(() => expect(screen.getByRole('note')).toHaveTextContent(/synthetic/i))
  })
})
