import { useState } from 'react'
import type { FormEvent } from 'react'

import type { OperatingMode, TransactionRequestInput, TransactionRiskResponse, TransactionType } from '../api'
import { TRANSACTION_TYPES, scoreTransaction } from '../api'
import { OperatingModeSelector } from '../components/OperatingModeSelector'
import { RiskBandBadge } from '../components/RiskBandBadge'
import { StatePanel } from '../components/StatePanel'
import { groupShapeErrorsByField } from '../lib/shapeErrors'

interface FormState {
  step: string
  type: TransactionType
  amount: string
  oldbalanceOrg: string
  newbalanceOrig: string
  oldbalanceDest: string
  newbalanceDest: string
}

const INITIAL_FORM: FormState = {
  step: '5',
  type: 'TRANSFER',
  amount: '',
  oldbalanceOrg: '',
  newbalanceOrig: '',
  oldbalanceDest: '',
  newbalanceDest: '',
}

type ViewState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'network_error'; message: string }
  | { kind: 'shape_error'; byField: Record<string, string[]>; general: string[] }
  | { kind: 'http_error'; status: number; detail: string }
  | { kind: 'result'; data: TransactionRiskResponse }

const NUMERIC_FIELDS: (keyof FormState)[] = [
  'amount',
  'oldbalanceOrg',
  'newbalanceOrig',
  'oldbalanceDest',
  'newbalanceDest',
]

const FIELD_LABELS: Record<keyof FormState, string> = {
  step: 'Simulated hour',
  type: 'Transaction type',
  amount: 'Transaction amount',
  oldbalanceOrg: 'Origin balance before',
  newbalanceOrig: 'Origin balance after',
  oldbalanceDest: 'Destination balance before',
  newbalanceDest: 'Destination balance after',
}

export function TransactionScoringPage() {
  const [form, setForm] = useState<FormState>(INITIAL_FORM)
  const [mode, setMode] = useState<OperatingMode>('BALANCED')
  const [formOpen, setFormOpen] = useState(false)
  const [view, setView] = useState<ViewState>({ kind: 'idle' })

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function chooseMode(nextMode: OperatingMode) {
    setMode(nextMode)
    setFormOpen(true)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setView({ kind: 'loading' })

    const payload: TransactionRequestInput = {
      step: Number(form.step),
      type: form.type,
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
    }

    const result = await scoreTransaction(payload, mode)
    if (result.kind === 'network_error') {
      setView(result)
    } else if (result.kind === 'shape_error') {
      const { byField, general } = groupShapeErrorsByField(result.errors)
      setView({ kind: 'shape_error', byField, general })
    } else if (result.kind === 'http_error') {
      setView({ kind: 'http_error', status: result.status, detail: result.detail })
    } else {
      setView({ kind: 'result', data: result.data })
    }
  }

  const fieldErrors: Record<string, string[]> = view.kind === 'shape_error' ? view.byField : {}

  return (
    <div>
      <h1>Merchant Risk Review</h1>
      <p className="muted">Find suspicious transactions before they become losses, then decide what deserves human review.</p>
      <h2>Transaction Scoring</h2>
      <p className="muted">Check whether one transaction looks safe, suspicious, or needs human review.</p>

      <div className="card">
        <OperatingModeSelector value={mode} onChange={chooseMode} />
      </div>

      {!formOpen && (
        <div className="card prompt-card">
          <h2>Start a transaction review</h2>
          <p className="muted">Choose a review preference above to enter transaction details.</p>
        </div>
      )}

      {formOpen && (
        <form className="card" onSubmit={handleSubmit}>
        <div className="field-row">
          <div className="field">
            <label htmlFor="field-step">{FIELD_LABELS.step}</label>
            <input
              id="field-step"
              aria-label="step"
              type="number"
              min={1}
              required
              value={form.step}
              onChange={(e) => updateField('step', e.target.value)}
              className={fieldErrors.step ? 'has-error' : ''}
            />
            <span className="field-hint">Enter an hour from the transaction history, such as 400</span>
            <FieldErrorText messages={fieldErrors.step} />
          </div>

          <div className="field">
            <label htmlFor="field-type">{FIELD_LABELS.type}</label>
            <select
              id="field-type"
              aria-label="type"
              value={form.type}
              onChange={(e) => updateField('type', e.target.value as TransactionType)}
              className={fieldErrors.type ? 'has-error' : ''}
            >
              {TRANSACTION_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <FieldErrorText messages={fieldErrors.type} />
          </div>

          {NUMERIC_FIELDS.map((key) => (
            <div className="field" key={key}>
              <label htmlFor={`field-${key}`}>{FIELD_LABELS[key]}</label>
              <input
                id={`field-${key}`}
                aria-label={key}
                type="number"
                min={0}
                step="any"
                required
                value={form[key]}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setForm((prev) => ({ ...prev, [key]: nextValue }))
                }}
                className={fieldErrors[key] ? 'has-error' : ''}
              />
              <FieldErrorText messages={fieldErrors[key]} />
            </div>
          ))}
        </div>

        {view.kind === 'shape_error' && view.general.length > 0 && (
          <p className="field-error">{view.general.join(' · ')}</p>
        )}

          <button className="btn-primary" type="submit" disabled={view.kind === 'loading'}>
            {view.kind === 'loading' ? 'Scoring…' : 'Score transaction'}
          </button>
        </form>
      )}

      <ResultView view={view} />
    </div>
  )
}

function FieldErrorText({ messages }: { messages?: string[] }) {
  if (!messages || messages.length === 0) return null
  return <span className="field-error">{messages.join(' · ')}</span>
}

function ResultView({ view }: { view: ViewState }) {
  if (view.kind === 'idle') return null

  if (view.kind === 'loading') {
    return <StatePanel kind="loading">Waiting for the API…</StatePanel>
  }

  if (view.kind === 'network_error') {
    return (
      <StatePanel kind="network-error">
        <p>{view.message}</p>
        <p className="muted">This is a connectivity problem, not a model decision - nothing was scored.</p>
      </StatePanel>
    )
  }

  if (view.kind === 'shape_error') {
    return (
      <StatePanel kind="invalid" title="Fix the highlighted field(s) above">
        <p>The request did not match the expected shape (checked before reaching the risk model).</p>
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
        {view.status === 401 && (
          <p className="muted">
            Check that the dashboard and risk service are using the same access key.
          </p>
        )}
      </StatePanel>
    )
  }

  const { data } = view

  if (data.status === 'invalid_input') {
    return (
      <StatePanel kind="invalid" title="Rejected: invalid transaction (domain rule)">
        <p>{data.message}</p>
        <p className="muted">Recommended fallback: {data.fallback}</p>
      </StatePanel>
    )
  }

  if (data.status === 'degraded') {
    return (
      <StatePanel kind="degraded" title="Model temporarily unavailable — MANUAL_REVIEW recommended">
        <p>{data.message}</p>
        <p className="muted">
          This is NOT a low-risk result. The scoring model or a required artifact is unavailable;
          no score was produced.
        </p>
      </StatePanel>
    )
  }

  // status === "ok"
  const riskBand = data.risk_band ?? 'LOW'

  return (
    <div className={`card result-card result-${riskBand.toLowerCase()}`}>
      <h2>Result</h2>
      <p>
        <RiskBandBadge band={riskBand} />
      </p>
      <table className="info-table">
        <tbody>
          <tr>
            <th>Risk score</th>
            <td>{data.risk_score?.toFixed(6)}</td>
          </tr>
          <tr>
            <th>Recommended action</th>
            <td>{data.recommended_action}</td>
          </tr>
          <tr>
            <th>Review preference</th>
            <td>
              {data.mode} (threshold {data.threshold_used})
            </td>
          </tr>
        </tbody>
      </table>

      <h3>Top reasons</h3>
      {data.top_reasons.length > 0 ? (
        <ul className="reasons-list">
          {data.top_reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">No reasons were returned by the model for this transaction.</p>
      )}

      {data.explanation?.narrative && <p className="muted">{data.explanation.narrative}</p>}

      {data.spike_context && (
        <p className="muted">
          Spike context for this step:{' '}
          {data.spike_context.available
            ? `${data.spike_context.is_spike ? 'flagged as an elevated-risk time window' : 'not currently flagged as a spike'} (see Spike Analysis for detail)`
            : 'unavailable'}
        </p>
      )}
    </div>
  )
}
