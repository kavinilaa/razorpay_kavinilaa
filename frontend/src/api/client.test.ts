import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * client.ts reads VITE_API_KEY at MODULE LOAD time, so tests that need a
 * specific key value must stub the env var and re-import a fresh module
 * instance (vi.resetModules) rather than mutating `import.meta.env` after
 * the fact.
 */
async function freshClient() {
  vi.resetModules()
  return import('./client')
}

function mockFetchOnce(response: { status: number; body: unknown }) {
  return vi.fn().mockResolvedValue({
    ok: response.status >= 200 && response.status < 300,
    status: response.status,
    json: async () => response.body,
  } as Response)
}

describe('api/client.ts', () => {
  beforeEach(() => {
    vi.unstubAllEnvs()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it('attaches the X-API-Key header from VITE_API_KEY to every request', async () => {
    vi.stubEnv('VITE_API_KEY', 'my-test-key')
    const fetchSpy = mockFetchOnce({ status: 200, body: { status: 'healthy' } })
    vi.stubGlobal('fetch', fetchSpy)

    const { apiRequest } = await freshClient()
    await apiRequest('/api/v1/health')

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [, init] = fetchSpy.mock.calls[0]
    expect((init.headers as Record<string, string>)['X-API-Key']).toBe('my-test-key')
  })

  it('sends no X-API-Key header when VITE_API_KEY is unset', async () => {
    vi.stubEnv('VITE_API_KEY', undefined)
    const fetchSpy = mockFetchOnce({ status: 200, body: { status: 'healthy' } })
    vi.stubGlobal('fetch', fetchSpy)

    const { apiRequest } = await freshClient()
    await apiRequest('/api/v1/health')

    const [, init] = fetchSpy.mock.calls[0]
    expect((init.headers as Record<string, string>)['X-API-Key']).toBeUndefined()
  })

  it('postForEnvelope treats a 401 as an http_error, NOT a success envelope', async () => {
    const fetchSpy = mockFetchOnce({ status: 401, body: { detail: 'Unauthorized' } })
    vi.stubGlobal('fetch', fetchSpy)

    const { postForEnvelope } = await freshClient()
    const result = await postForEnvelope('/api/v1/transactions/score', { step: 1 })

    expect(result.kind).toBe('http_error')
    if (result.kind === 'http_error') {
      expect(result.status).toBe(401)
      expect(result.detail).toBe('Unauthorized')
    }
  })

  it('postForEnvelope still treats a real risk-engine envelope (with a "status" field) as success, even at 422', async () => {
    const fetchSpy = mockFetchOnce({
      status: 422,
      body: { status: 'invalid_input', risk_score: null, message: 'bad type', fallback: 'MANUAL_REVIEW' },
    })
    vi.stubGlobal('fetch', fetchSpy)

    const { postForEnvelope } = await freshClient()
    const result = await postForEnvelope('/api/v1/transactions/score', { step: 1 })

    expect(result.kind).toBe('success')
    if (result.kind === 'success') {
      expect((result.data as { status: string }).status).toBe('invalid_input')
    }
  })

  it('postForEnvelope still recognizes a Pydantic shape-validation error body over a 401-shaped one', async () => {
    const fetchSpy = mockFetchOnce({
      status: 422,
      body: { detail: [{ type: 'missing', loc: ['body', 'amount'], msg: 'Field required' }] },
    })
    vi.stubGlobal('fetch', fetchSpy)

    const { postForEnvelope } = await freshClient()
    const result = await postForEnvelope('/api/v1/transactions/score', { step: 1 })

    expect(result.kind).toBe('shape_error')
  })
})
