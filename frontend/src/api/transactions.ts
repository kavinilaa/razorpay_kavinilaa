import type { HttpErrorOutcome, NetworkErrorOutcome, ShapeErrorOutcome, SuccessOutcome } from './client'
import { postForEnvelope } from './client'
import type { OperatingMode, TransactionRequestInput, TransactionRiskResponse } from './types'

export type ScoreTransactionOutcome =
  | SuccessOutcome<TransactionRiskResponse>
  | ShapeErrorOutcome
  | HttpErrorOutcome
  | NetworkErrorOutcome

/**
 * POST /api/v1/transactions/score
 *
 * The returned `data.status` is one of "ok" | "invalid_input" | "degraded" -
 * ALL THREE arrive as `kind: "success"` here, because all three are
 * well-formed HTTP responses from the risk engine (200 or 422, but always
 * carrying the same RiskAssessment-shaped body). Only a genuine transport
 * failure (`network_error`) or FastAPI's own pre-handler shape-validation
 * rejection (`shape_error`) short-circuit before reaching that envelope.
 * See src/risk_engine/schemas.py::RiskAssessment and
 * reports/phase5_api_summary.md for why this envelope carries all three
 * business outcomes uniformly.
 *
 * Phase 7: a missing/wrong API key short-circuits as `http_error` (401)
 * BEFORE the request ever reaches predict_transaction_risk() - a fourth,
 * distinct outcome from all three business statuses above.
 */
export async function scoreTransaction(
  payload: TransactionRequestInput,
  mode: OperatingMode,
): Promise<ScoreTransactionOutcome> {
  const query = new URLSearchParams({ operating_mode: mode }).toString()
  return postForEnvelope<TransactionRiskResponse>(`/api/v1/transactions/score?${query}`, payload)
}
