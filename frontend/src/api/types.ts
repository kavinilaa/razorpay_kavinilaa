/**
 * Phase 6 - TypeScript types mirroring the Phase 5 backend's Pydantic schemas.
 *
 * These are kept in sync BY HAND with:
 *   - backend/schemas/transaction.py (TransactionRequest / TransactionRiskResponse,
 *     themselves generated from src/risk_engine/schemas.py - see that file's docstring)
 *   - backend/schemas/spike.py (SpikeAnalysisRequest / SpikeAnalysisResponse)
 *   - backend/api/health.py::run_health_check()
 *   - backend/api/model_info.py::build_model_info()
 *
 * No field here should be invented or renamed relative to those sources - if the
 * backend response shape changes, this file must change with it (see
 * reports/phase6_frontend_summary.md for why this isn't code-generated in this phase).
 */

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/** The three operating modes defined in src/risk_engine/thresholds.py. */
export type OperatingMode = 'HIGH_RECALL' | 'BALANCED' | 'HIGH_PRECISION'

export const OPERATING_MODES: OperatingMode[] = ['HIGH_RECALL', 'BALANCED', 'HIGH_PRECISION']

/** The five PaySim transaction types risk_engine.schemas.VALID_TYPES recognizes. */
export type TransactionType = 'PAYMENT' | 'TRANSFER' | 'CASH_OUT' | 'CASH_IN' | 'DEBIT'

export const TRANSACTION_TYPES: TransactionType[] = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']

/** FastAPI's own automatic validation-error shape (a Pydantic shape-validation
 * failure caught BEFORE the request reaches predict_transaction_risk() /
 * analyze_spike()) - distinct from either endpoint's own "status" envelope. */
export interface PydanticValidationError {
  type: string
  loc: (string | number)[]
  msg: string
  input?: unknown
  ctx?: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// POST /api/v1/transactions/score
// ---------------------------------------------------------------------------

export interface TransactionRequestInput {
  step: number
  type: TransactionType
  amount: number
  oldbalanceOrg: number
  newbalanceOrig: number
  oldbalanceDest: number
  newbalanceDest: number
  // Optional destination-history overrides (src/risk_engine/schemas.py OPTIONAL_FIELDS_DEFAULTS).
  // Omitted in this phase's form - the risk engine defaults them to "brand new destination".
  nameOrig?: string
  nameDest?: string
  hist_dest_txn_count?: number
  hist_dest_unique_origin_count?: number
  hist_dest_total_amount?: number
  hist_dest_avg_amount?: number
  hist_dest_max_amount?: number
}

export interface Contributor {
  feature: string
  value: number | string
  shap_value: number
  phrase: string
}

export interface Explanation {
  top_positive_contributors: Contributor[]
  top_negative_contributors: Contributor[]
  feature_values: Record<string, number | string>
  narrative: string
}

export interface SpikeContext {
  available: boolean
  is_spike?: boolean
  statistical_method_flag?: boolean
  isolation_forest_flag?: boolean
  txn_count_vs_hour_baseline_ratio?: number
  predicted_high_risk_count_vs_hour_baseline_ratio?: number
  note?: string
  message?: string
}

/** Mirrors risk_engine.schemas.RiskAssessment exactly - the shape
 * predict_transaction_risk() returns for status "ok", "invalid_input", AND
 * "degraded" alike (only the field values differ). */
export interface TransactionRiskResponse {
  status: 'ok' | 'invalid_input' | 'degraded'
  risk_score: number | null
  risk_band: 'LOW' | 'MEDIUM' | 'HIGH' | null
  recommended_action: string | null
  mode: OperatingMode | string | null
  threshold_used: number | null
  top_reasons: string[]
  explanation: Explanation | null
  spike_context: SpikeContext | null
  message: string | null
  fallback: string | null
}

// ---------------------------------------------------------------------------
// POST /api/v1/spikes/analyze
// ---------------------------------------------------------------------------

export interface SpikeAnalysisRequestInput {
  step?: number
  step_aggregate?: Record<string, unknown>
  transactions_in_step?: Array<{ nameDest: string; amount: number }>
}

export interface RootCauseContributor {
  factor: string
  ratio_vs_baseline: number | null
  actual: number | null
  baseline: number | null
}

export interface RootCauseResult {
  step: number | null
  hour_of_day: number | null
  primary_driver: string
  contributors: RootCauseContributor[]
  baseline_comparison: Record<string, unknown>
  evidence: string[]
}

export interface SpikeAnalysisResponse {
  step?: number
  spike_context?: SpikeContext
  root_cause?: RootCauseResult
}

// ---------------------------------------------------------------------------
// GET /api/v1/health
// ---------------------------------------------------------------------------

export interface HealthComponent {
  loaded: boolean
  path: string
  error: string | null
}

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  components: {
    transaction_model: HealthComponent
    spike_detector: HealthComponent
    feature_reference_stats: HealthComponent
  }
  notes?: Record<string, string>
  checked_at: string
}

// ---------------------------------------------------------------------------
// GET /api/v1/model-info
// ---------------------------------------------------------------------------

export interface OperatingModeMetrics {
  precision: number
  recall: number
  f1?: number
  alert_rate: number | null
}

export interface OperatingModeConfig {
  threshold: number
  validation: OperatingModeMetrics
  test: OperatingModeMetrics
}

export interface ModelMetrics {
  pr_auc: number
  roc_auc: number
  precision: number
  recall: number
}

export interface ModelInfoResponse {
  model_name: string
  model_version: string
  model_file_sha256: string | null
  random_seed: number
  train_steps: [number, number]
  val_steps: [number, number]
  test_steps: [number, number]
  modeling_universe: string
  target: string
  feature_count: number
  features: string[]
  validation_metrics: ModelMetrics | null
  test_metrics: ModelMetrics | null
  operating_modes: Record<OperatingMode, OperatingModeConfig>
  data_provenance: string
  synthetic_data_caveat: string
  prohibited_use: string
  source_artifacts: Record<string, string>
}
