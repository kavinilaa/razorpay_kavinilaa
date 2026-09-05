import { useEffect, useState } from 'react'

import type { ApiOutcome, HealthResponse, ModelInfoResponse } from '../api'
import { fetchHealth } from '../api'
import { StatePanel } from '../components/StatePanel'
import { useModelInfo } from '../context/ModelInfoContext'

export function ModelInfoPage() {
  const { outcome: modelInfoOutcome, refetch } = useModelInfo()
  const [healthOutcome, setHealthOutcome] = useState<ApiOutcome<HealthResponse> | { kind: 'loading' }>({
    kind: 'loading',
  })

  useEffect(() => {
    let cancelled = false
    setHealthOutcome({ kind: 'loading' })
    fetchHealth().then((result) => {
      if (!cancelled) setHealthOutcome(result)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div>
      <h1>Model Info &amp; Health</h1>

      <div className="card">
        <h2>System status</h2>
        <HealthSection outcome={healthOutcome} />
      </div>

      <div className="card">
        <h2>Scoring system details</h2>
        <ModelInfoSection outcome={modelInfoOutcome} onRetry={refetch} />
      </div>
    </div>
  )
}

function HealthSection({ outcome }: { outcome: ApiOutcome<HealthResponse> | { kind: 'loading' } }) {
  if (outcome.kind === 'loading') return <StatePanel kind="loading">Checking health…</StatePanel>
  if (outcome.kind === 'network_error') {
    return (
      <StatePanel kind="network-error">
        <p>{outcome.message}</p>
      </StatePanel>
    )
  }
  if (outcome.kind === 'http_error') {
    return (
      <StatePanel kind="degraded" title={`Health check failed (HTTP ${outcome.status})`}>
        <p>{outcome.detail}</p>
      </StatePanel>
    )
  }

  const health = outcome.data
  return (
    <div>
      <p>
        <span className={`health-dot ${health.status === 'healthy' ? 'ok' : 'fail'}`} aria-hidden="true" />
        Overall status: <strong>{health.status.toUpperCase()}</strong>
      </p>
      <table className="info-table">
        <thead>
          <tr>
            <th>Component</th>
            <th>Loaded</th>
            <th>Path</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(health.components).map(([name, component]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>
                <span className={`health-dot ${component.loaded ? 'ok' : 'fail'}`} aria-hidden="true" />
                {component.loaded ? 'yes' : 'no'}
              </td>
              <td style={{ fontSize: 11, wordBreak: 'break-all' }}>{component.path}</td>
              <td>{component.error ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Checked at: {health.checked_at}</p>
    </div>
  )
}

function ModelInfoSection({
  outcome,
  onRetry,
}: {
  outcome: ApiOutcome<ModelInfoResponse> | { kind: 'loading' }
  onRetry: () => void
}) {
  if (outcome.kind === 'loading') return <StatePanel kind="loading">Loading model info…</StatePanel>
  if (outcome.kind === 'network_error') {
    return (
      <StatePanel kind="network-error">
        <p>{outcome.message}</p>
        <button className="btn-primary" onClick={onRetry} type="button">
          Retry
        </button>
      </StatePanel>
    )
  }
  if (outcome.kind === 'http_error') {
    const isAuthError = outcome.status === 401
    return (
      <StatePanel
        kind={isAuthError ? 'auth-error' : 'degraded'}
        title={isAuthError ? 'Authentication failed' : `Model info unavailable (HTTP ${outcome.status})`}
      >
        <p>{outcome.detail}</p>
        {isAuthError && (
          <p className="muted">
            Check that VITE_API_KEY (frontend) matches RISK_API_API_KEY (backend).
          </p>
        )}
        <button className="btn-primary" onClick={onRetry} type="button">
          Retry
        </button>
      </StatePanel>
    )
  }

  const info = outcome.data

  return (
    <div>
      <p>
        <strong>Data provenance:</strong> {info.data_provenance}
      </p>
      <p className="muted">{info.synthetic_data_caveat}</p>

      <table className="info-table">
        <tbody>
          <tr>
            <th>Model</th>
            <td>{info.model_name}</td>
          </tr>
          <tr>
            <th>Version</th>
            <td>{info.model_version}</td>
          </tr>
          <tr>
            <th>SHA-256</th>
            <td style={{ fontSize: 11, wordBreak: 'break-all' }}>{info.model_file_sha256}</td>
          </tr>
          <tr>
              <th>Evaluation periods</th>
            <td>
              {info.train_steps.join('–')} / {info.val_steps.join('–')} / {info.test_steps.join('–')}
            </td>
          </tr>
          <tr>
            <th>Modeling universe</th>
            <td>{info.modeling_universe}</td>
          </tr>
          <tr>
            <th>Target</th>
            <td>{info.target}</td>
          </tr>
          <tr>
            <th>Feature count</th>
            <td>{info.feature_count}</td>
          </tr>
          {info.validation_metrics && (
            <tr>
              <th>Validation quality</th>
              <td>
                {info.validation_metrics.pr_auc.toFixed(4)} / {info.validation_metrics.roc_auc.toFixed(4)}
              </td>
            </tr>
          )}
          {info.test_metrics && (
            <tr>
              <th>Held-out test quality</th>
              <td>
                {info.test_metrics.pr_auc.toFixed(4)} / {info.test_metrics.roc_auc.toFixed(4)}
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <h3>Review preferences</h3>
      <table className="info-table">
        <thead>
          <tr>
            <th>Mode</th>
            <th>Threshold</th>
            <th>Val precision</th>
            <th>Val recall</th>
            <th>Val alert rate</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(info.operating_modes).map(([mode, cfg]) => (
            <tr key={mode}>
              <td>{mode}</td>
              <td>{cfg.threshold}</td>
              <td>{(cfg.validation.precision * 100).toFixed(1)}%</td>
              <td>{(cfg.validation.recall * 100).toFixed(1)}%</td>
              <td>{cfg.validation.alert_rate != null ? `${(cfg.validation.alert_rate * 100).toFixed(2)}%` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Prohibited use</h3>
      <p className="muted">{info.prohibited_use}</p>
    </div>
  )
}
