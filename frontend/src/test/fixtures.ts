import type {
  HealthResponse,
  ModelInfoResponse,
  SpikeAnalysisResponse,
  TransactionRiskResponse,
} from '../api'

export const MODEL_INFO_FIXTURE: ModelInfoResponse = {
  model_name: 'xgboost',
  model_version: 'xgboost-seed42-abc123',
  model_file_sha256: 'a'.repeat(64),
  random_seed: 42,
  train_steps: [1, 520],
  val_steps: [521, 631],
  test_steps: [632, 743],
  modeling_universe: 'TRANSFER + CASH_OUT only',
  target: 'isFraud',
  feature_count: 23,
  features: ['amount', 'drained_to_zero'],
  validation_metrics: { pr_auc: 0.9993, roc_auc: 1.0, precision: 1.0, recall: 0.9992 },
  test_metrics: { pr_auc: 1.0, roc_auc: 1.0, precision: 1.0, recall: 0.9992 },
  operating_modes: {
    HIGH_RECALL: { threshold: 0.43, validation: { precision: 1.0, recall: 0.9992, alert_rate: 0.015 }, test: { precision: 1.0, recall: 0.9992, alert_rate: null } },
    BALANCED: { threshold: 0.43, validation: { precision: 1.0, recall: 0.9992, alert_rate: 0.015 }, test: { precision: 1.0, recall: 0.9992, alert_rate: null } },
    HIGH_PRECISION: { threshold: 0.17, validation: { precision: 0.9, recall: 0.9992, alert_rate: 0.0166 }, test: { precision: 0.965, recall: 1.0, alert_rate: null } },
  },
  data_provenance: 'synthetic PaySim dataset; not validated on real transaction data',
  synthetic_data_caveat: 'Fraud in this dataset almost always drains the origin account balance to zero.',
  prohibited_use: 'Must not be presented as a production-ready fraud detector trained on real data.',
  source_artifacts: { model_metadata: 'models/model_metadata.json' },
}

export const HEALTHY_FIXTURE: HealthResponse = {
  status: 'healthy',
  components: {
    transaction_model: { loaded: true, path: 'models/xgboost.joblib', error: null },
    spike_detector: { loaded: true, path: 'data/processed/time_window_features_extended.parquet', error: null },
    feature_reference_stats: { loaded: true, path: 'models/feature_reference_stats.json', error: null },
  },
  checked_at: '2026-01-01T00:00:00Z',
}

export const OK_RESULT_FIXTURE: TransactionRiskResponse = {
  status: 'ok',
  risk_score: 0.895437,
  risk_band: 'HIGH',
  recommended_action: 'REVIEW',
  mode: 'BALANCED',
  threshold_used: 0.43,
  top_reasons: [
    'the origin balance exactly matches the expected debit formula (oldbalance - amount = newbalance)',
    'the transaction amount is Rs.1,500,000',
  ],
  explanation: {
    top_positive_contributors: [
      { feature: 'balance_error_orig', value: 0, shap_value: 1.48, phrase: 'the origin balance exactly matches the expected debit formula' },
    ],
    top_negative_contributors: [],
    feature_values: { amount: 1500000 },
    narrative: 'High risk (score 0.895) because the origin balance exactly matches the expected debit formula.',
  },
  spike_context: {
    available: true,
    is_spike: false,
    statistical_method_flag: false,
    isolation_forest_flag: false,
  },
  message: null,
  fallback: null,
}

export const INVALID_INPUT_RESULT_FIXTURE: TransactionRiskResponse = {
  status: 'invalid_input',
  risk_score: null,
  risk_band: null,
  recommended_action: null,
  mode: 'BALANCED',
  threshold_used: null,
  top_reasons: [],
  explanation: null,
  spike_context: null,
  message: "Transaction failed validation: unknown transaction type: 'NOT_A_TYPE' (must be one of ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])",
  fallback: 'MANUAL_REVIEW',
}

export const DEGRADED_RESULT_FIXTURE: TransactionRiskResponse = {
  status: 'degraded',
  risk_score: null,
  risk_band: null,
  recommended_action: 'MANUAL_REVIEW',
  mode: 'BALANCED',
  threshold_used: null,
  top_reasons: [],
  explanation: null,
  spike_context: null,
  message: 'Transaction risk model unavailable: model file not found at models/xgboost.joblib',
  fallback: 'MANUAL_REVIEW',
}

export const SPIKE_LOOKUP_FIXTURE: SpikeAnalysisResponse = {
  step: 5,
  spike_context: {
    available: true,
    is_spike: false,
    statistical_method_flag: false,
    isolation_forest_flag: false,
    txn_count_vs_hour_baseline_ratio: 0.55,
    predicted_high_risk_count_vs_hour_baseline_ratio: 0.08,
  },
}

export const SPIKE_UNAVAILABLE_FIXTURE: SpikeAnalysisResponse = {
  step: 5,
  spike_context: {
    available: false,
    message: 'spike-detection table not found at ... - run src/models/spike_detection.py first',
  },
}

export const ROOT_CAUSE_NO_DRIVER_FIXTURE: SpikeAnalysisResponse = {
  root_cause: {
    step: 633,
    hour_of_day: 9,
    primary_driver: 'no single driver identified',
    contributors: [],
    baseline_comparison: {},
    evidence: [
      'No individual factor exceeded the driver threshold (1.5x baseline / 30% destination share).',
    ],
  },
}
