import { useState } from 'react'
import type { FormEvent } from 'react'

import type { SpikeAnalysisResponse } from '../api'
import { analyzeSpike } from '../api'
import { StatePanel } from '../components/StatePanel'
import { groupShapeErrorsByField } from '../lib/shapeErrors'

type ViewState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'network_error'; message: string }
  | { kind: 'shape_error'; general: string[] }
  | { kind: 'http_error'; status: number; detail: string }
  | { kind: 'result'; data: SpikeAnalysisResponse }

const AGGREGATE_FIELDS = [
  'step',
  'hour_of_day',
  'transaction_count',
  'hist_avg_txn_count_same_hour',
  'transfer_count',
  'hist_avg_transfer_count_same_hour',
  'cash_out_count',
  'hist_avg_cash_out_count_same_hour',
  'high_amount_count',
  'hist_avg_high_amount_count_same_hour',
  'predicted_high_risk_count',
  'hist_avg_predicted_high_risk_count_same_hour',
] as const

const AGGREGATE_LABELS: Record<(typeof AGGREGATE_FIELDS)[number], string> = {
  step: 'Simulated hour',
  hour_of_day: 'Hour of day',
  transaction_count: 'Total transactions',
  hist_avg_txn_count_same_hour: 'Usual transactions for this hour',
  transfer_count: 'Transfers',
  hist_avg_transfer_count_same_hour: 'Usual transfers for this hour',
  cash_out_count: 'Cash-outs',
  hist_avg_cash_out_count_same_hour: 'Usual cash-outs for this hour',
  high_amount_count: 'High-value transactions',
  hist_avg_high_amount_count_same_hour: 'Usual high-value transactions',
  predicted_high_risk_count: 'High-risk transactions',
  hist_avg_predicted_high_risk_count_same_hour: 'Usual high-risk transactions',
}

export function SpikeAnalysisPage() {
  const [stepLookup, setStepLookup] = useState('')
  const [aggregate, setAggregate] = useState<Record<string, string>>({})
  const [useAggregate, setUseAggregate] = useState(false)
  const [view, setView] = useState<ViewState>({ kind: 'idle' })

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setView({ kind: 'loading' })

    const payload: { step?: number; step_aggregate?: Record<string, number> } = {}
    if (stepLookup.trim() !== '') {
      payload.step = Number(stepLookup)
    }
    if (useAggregate) {
      const numericAggregate: Record<string, number> = {}
      for (const field of AGGREGATE_FIELDS) {
        const raw = aggregate[field]
        if (raw !== undefined && raw !== '') numericAggregate[field] = Number(raw)
      }
      payload.step_aggregate = numericAggregate
    }

    const result = await analyzeSpike(payload)
    if (result.kind === 'network_error') {
      setView(result)
    } else if (result.kind === 'shape_error') {
      const { general, byField } = groupShapeErrorsByField(result.errors)
      setView({ kind: 'shape_error', general: [...general, ...Object.values(byField).flat()] })
    } else if (result.kind === 'http_error') {
      setView({ kind: 'http_error', status: result.status, detail: result.detail })
    } else {
      setView({ kind: 'result', data: result.data })
    }
  }

  return (
    <div>
      <h1>Spike Analysis</h1>
      <p className="muted">
        Check whether activity during a simulated hour is unusual and see what may have driven it.
      </p>

      <form className="card" onSubmit={handleSubmit}>
        <h3>Check one simulated hour</h3>
        <div className="field-row">
          <div className="field">
            <label htmlFor="step-lookup">Simulated hour</label>
            <input
              id="step-lookup"
              aria-label="step"
              type="number"
              min={1}
              value={stepLookup}
              onChange={(e) => setStepLookup(e.target.value)}
              placeholder="e.g. 633"
            />
            <span className="field-hint">Enter an hour from the transaction history, such as 633</span>
          </div>
        </div>

        <h3>
          <label>
            <input
              type="checkbox"
              checked={useAggregate}
              onChange={(e) => setUseAggregate(e.target.checked)}
            />{' '}
            Also investigate what drove unusual activity
            <span className="visually-hidden"> Also run root-cause analysis on a step aggregate</span>
          </label>
        </h3>
        {useAggregate && (
          <div className="field-row">
            {AGGREGATE_FIELDS.map((field) => (
              <div className="field" key={field}>
                <label htmlFor={`agg-${field}`}>{AGGREGATE_LABELS[field]}</label>
                <input
                  id={`agg-${field}`}
                  aria-label={field}
                  type="number"
                  value={aggregate[field] ?? ''}
                  onChange={(e) => setAggregate((prev) => ({ ...prev, [field]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        )}

        <button className="btn-primary" type="submit" disabled={view.kind === 'loading'}>
          {view.kind === 'loading' ? 'Analyzing…' : 'Analyze'}
        </button>
      </form>

      <ResultView view={view} />
    </div>
  )
}

function ResultView({ view }: { view: ViewState }) {
  if (view.kind === 'idle') return null
  if (view.kind === 'loading') return <StatePanel kind="loading">Waiting for the API…</StatePanel>
  if (view.kind === 'network_error') {
    return (
      <StatePanel kind="network-error">
        <p>{view.message}</p>
      </StatePanel>
    )
  }
  if (view.kind === 'shape_error') {
    return (
      <StatePanel kind="invalid" title="Invalid request">
        <p>{view.general.join(' · ') || 'Provide either a step or a step aggregate.'}</p>
      </StatePanel>
    )
  }

  if (view.kind === 'http_error') {
    return (
      <StatePanel
        kind={view.status === 401 ? 'auth-error' : 'degraded'}
        title={view.status === 401 ? 'Authentication failed' : `Request failed (HTTP ${view.status})`}
      >
        <p>{view.detail}</p>
      </StatePanel>
    )
  }

  const { data } = view

  return (
    <div>
      {data.spike_context !== undefined && (
        <div className={`card activity-card activity-${data.spike_context.is_spike ? 'flagged' : 'clear'}`}>
          <h2>Activity check (hour {data.step})</h2>
          {data.spike_context.available ? (
            <>
              <p>
                Overall activity flag:{' '}
                <strong>{data.spike_context.is_spike ? 'FLAGGED' : 'not flagged'}</strong>
              </p>
              <div className="signal-row">
                <div
                  className={`signal-chip flag-${data.spike_context.statistical_method_flag ? 'true' : 'false'}`}
                >
                  <span className="signal-name">Volume pattern check</span>
                  <span className="visually-hidden">Statistical (z-score) signal</span>
                  <span className="signal-value">
                    {data.spike_context.statistical_method_flag ? 'FLAGGED' : 'not flagged'}
                  </span>
                </div>
                <div
                  className={`signal-chip flag-${data.spike_context.isolation_forest_flag ? 'true' : 'false'}`}
                >
                  <span className="signal-name">Unusual activity check</span>
                  <span className="visually-hidden">Isolation Forest signal</span>
                  <span className="signal-value">
                    {data.spike_context.isolation_forest_flag ? 'FLAGGED' : 'not flagged'}
                  </span>
                </div>
              </div>
              <p className="muted">
                This is a supporting signal for human review, not an automatic decision.
              </p>
            </>
          ) : (
            <StatePanel kind="degraded" title="Spike detector unavailable">
              <p>{data.spike_context.message ?? data.spike_context.note}</p>
            </StatePanel>
          )}
        </div>
      )}

      {data.root_cause && (
        <div className="card">
          <h2>What may have driven the alert?</h2>
          <p>
            Main reason: <strong>{data.root_cause.primary_driver}</strong>
          </p>
          {data.root_cause.primary_driver === 'no single driver identified' && (
            <p className="muted">
              No single activity type was unusually high during this hour. This is a finding, not
              missing information.
              <span className="visually-hidden">
                No single transaction-type driver exceeded its historical baseline for this step.
              </span>
            </p>
          )}
          {data.root_cause.contributors.length > 0 && (
            <table className="info-table">
              <thead>
                <tr>
                  <th>Factor</th>
                  <th>Compared with usual</th>
                  <th>This hour</th>
                  <th>Usual level</th>
                </tr>
              </thead>
              <tbody>
                {data.root_cause.contributors.map((c, i) => (
                  <tr key={i}>
                    <td>{c.factor}</td>
                    <td>{c.ratio_vs_baseline?.toFixed(2) ?? '—'}</td>
                    <td>{c.actual ?? '—'}</td>
                    <td>{c.baseline ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <h3>Evidence</h3>
          <ul className="reasons-list">
            {data.root_cause.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
