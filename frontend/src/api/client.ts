/**
 * Phase 6 - shared fetch plumbing for the 4 backend endpoints.
 *
 * Every wrapper in api/transactions.ts, api/spikes.ts, api/health.ts,
 * api/modelInfo.ts returns a discriminated union so callers (and tests)
 * are FORCED to handle a genuine network/timeout failure differently from
 * a well-formed API response - including one whose body says
 * status="invalid_input" or status="degraded". That distinction is a
 * cross-cutting requirement of this phase, not incidental.
 *
 * Phase 7 addition: every request attaches the shared-secret API key (from
 * VITE_API_KEY, see .env.example) under the `X-API-Key` header - this must
 * match backend/core/config.py's `api_key_header_name` default. It is sent
 * on EVERY request, including GET /health, for simplicity: /health ignores
 * auth entirely (backend/core/auth.py), so an extra header there is
 * harmless. A missing/wrong key surfaces as its own `http_error` outcome
 * (status 401) - see `postForEnvelope` below, which must NOT mistake a 401's
 * `{ "detail": "Unauthorized" }` body for a risk-engine envelope.
 */
import type { PydanticValidationError } from './types'

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/** Must match backend/core/config.py's `api_key_header_name` default. Not
 * itself read from an env var on the frontend - only the KEY VALUE
 * (VITE_API_KEY) is - see reports/phase7_summary.md for why the header name
 * is treated as a fixed API contract rather than independently configurable
 * on both sides. */
const API_KEY_HEADER_NAME = 'X-API-Key'

const API_KEY: string | undefined = import.meta.env.VITE_API_KEY

const REQUEST_TIMEOUT_MS = 10_000

/** A genuine transport-level failure: the request never got a response at
 * all (offline, DNS failure, connection refused, timeout). This is NOT the
 * same thing as the API responding with status="degraded" - that's a
 * successful HTTP response describing a handled backend problem. */
export interface NetworkErrorOutcome {
  kind: 'network_error'
  message: string
}

/** FastAPI's own automatic Pydantic shape-validation failure (422, body
 * shape `{ detail: [...] }`) - caught before the request ever reached the
 * risk engine. Distinct from the risk engine's own status="invalid_input"
 * envelope, which has a completely different body shape (see
 * reports/phase5_api_summary.md). */
export interface ShapeErrorOutcome {
  kind: 'shape_error'
  errors: PydanticValidationError[]
}

/** A non-2xx response that is neither of the above (e.g. GET /model-info's
 * 503 when models/model_metadata.json is missing). */
export interface HttpErrorOutcome {
  kind: 'http_error'
  status: number
  detail: string
}

export interface SuccessOutcome<T> {
  kind: 'success'
  data: T
}

export type ApiOutcome<T> = SuccessOutcome<T> | HttpErrorOutcome | NetworkErrorOutcome

function isPydanticShapeErrorBody(body: unknown): body is { detail: PydanticValidationError[] } {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) return false
  const detail = (body as Record<string, unknown>).detail
  return (
    Array.isArray(detail) &&
    detail.length > 0 &&
    detail.every((d) => typeof d === 'object' && d !== null && 'loc' in d && 'msg' in d)
  )
}

async function doFetch(path: string, init?: RequestInit): Promise<Response | NetworkErrorOutcome> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  const headers: Record<string, string> = {
    ...(API_KEY ? { [API_KEY_HEADER_NAME]: API_KEY } : {}),
    ...((init?.headers as Record<string, string> | undefined) ?? {}),
  }
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...init, headers, signal: controller.signal })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { kind: 'network_error', message: `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s.` }
    }
    const message = err instanceof Error ? err.message : 'Network request failed.'
    return { kind: 'network_error', message: `Could not reach the API: ${message}` }
  } finally {
    clearTimeout(timeoutId)
  }
}

/**
 * GET/POST helper for the two "always JSON-body-shaped" endpoints
 * (health, model-info) where non-2xx just means "something went wrong",
 * with no risk-engine-specific envelope to distinguish.
 */
export async function apiRequest<T>(path: string, init?: RequestInit): Promise<ApiOutcome<T>> {
  const response = await doFetch(path, init)
  if ('kind' in response) return response

  let body: unknown
  try {
    body = await response.json()
  } catch {
    return {
      kind: 'network_error',
      message: `The API returned a response that could not be parsed as JSON (status ${response.status}).`,
    }
  }

  if (response.ok) {
    return { kind: 'success', data: body as T }
  }

  const detail =
    typeof body === 'object' && body !== null && 'detail' in body && typeof (body as Record<string, unknown>).detail === 'string'
      ? ((body as Record<string, unknown>).detail as string)
      : `Request failed with status ${response.status}.`
  return { kind: 'http_error', status: response.status, detail }
}

/**
 * POST helper for the two endpoints (transactions/score, spikes/analyze)
 * whose 200/422 responses both carry the risk-engine's OWN envelope shape
 * (a "status"-bearing object for scoring; a plain result object for spike
 * analysis) - and whose 422s can ALSO be FastAPI's unrelated automatic
 * shape-validation error, or (Phase 7) a 401 from `require_api_key` with
 * body `{ "detail": "Unauthorized" }`. That 401 body must NOT be mistaken
 * for a risk-engine envelope (it has no "status" field, unlike every real
 * envelope response, which always includes one - even status="invalid_input"
 * at 422): if it were, a 401 would be silently rendered as if it were a
 * `TransactionRiskResponse`/`SpikeAnalysisResponse` with every field
 * `undefined`, rather than the distinct "you're not authenticated" state
 * this deliberately surfaces instead.
 */
export async function postForEnvelope<T>(
  path: string,
  payload: unknown,
): Promise<SuccessOutcome<T> | ShapeErrorOutcome | HttpErrorOutcome | NetworkErrorOutcome> {
  const response = await doFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if ('kind' in response) return response

  let body: unknown
  try {
    body = await response.json()
  } catch {
    return {
      kind: 'network_error',
      message: `The API returned a response that could not be parsed as JSON (status ${response.status}).`,
    }
  }

  if (isPydanticShapeErrorBody(body)) {
    return { kind: 'shape_error', errors: body.detail }
  }

  const isRiskEngineEnvelope = typeof body === 'object' && body !== null && 'status' in body
  if (!response.ok && !isRiskEngineEnvelope) {
    const detail =
      typeof body === 'object' && body !== null && typeof (body as Record<string, unknown>).detail === 'string'
        ? ((body as Record<string, unknown>).detail as string)
        : `Request failed with status ${response.status}.`
    return { kind: 'http_error', status: response.status, detail }
  }

  return { kind: 'success', data: body as T }
}
