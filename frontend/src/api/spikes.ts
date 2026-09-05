import type { HttpErrorOutcome, NetworkErrorOutcome, ShapeErrorOutcome, SuccessOutcome } from './client'
import { postForEnvelope } from './client'
import type { SpikeAnalysisRequestInput, SpikeAnalysisResponse } from './types'

export type AnalyzeSpikeOutcome =
  | SuccessOutcome<SpikeAnalysisResponse>
  | ShapeErrorOutcome
  | HttpErrorOutcome
  | NetworkErrorOutcome

/** POST /api/v1/spikes/analyze - see backend/api/spike.py. Always 200 on a
 * well-formed, AUTHENTICATED request (spike_context.available=false
 * communicates an unavailable detector without failing the call); 422 only
 * for a shape-invalid request (neither `step` nor `step_aggregate` supplied,
 * or a negative step). Phase 7: a missing/wrong API key short-circuits as
 * `http_error` (401) before the request reaches this endpoint's own logic. */
export async function analyzeSpike(payload: SpikeAnalysisRequestInput): Promise<AnalyzeSpikeOutcome> {
  return postForEnvelope<SpikeAnalysisResponse>('/api/v1/spikes/analyze', payload)
}
