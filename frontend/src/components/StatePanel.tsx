import type { ReactNode } from 'react'

export type StateKind = 'loading' | 'invalid' | 'degraded' | 'network-error' | 'auth-error'

const GLYPH: Record<StateKind, string> = {
  loading: '…',
  invalid: '⚠',
  degraded: '⛔',
  'network-error': '⚡',
  'auth-error': '🔒',
}

const TITLE: Record<StateKind, string> = {
  loading: 'Loading…',
  invalid: 'Invalid request',
  degraded: 'Model temporarily unavailable',
  'network-error': 'Could not reach the server',
  'auth-error': 'Authentication failed',
}

/**
 * A response-state panel visually distinct from a risk-band result AND from
 * the other four states (see index.css: each state uses a different color
 * family and/or background pattern - "degraded" uses a hazard-stripe pattern
 * specifically so it can never be mistaken for a calm, successful low-risk
 * result). This is the ONE place all five non-happy-path states are
 * rendered, so pages stay consistent.
 *
 * "auth-error" (Phase 7) is deliberately its own kind, not folded into
 * "network-error" or "degraded": the server IS reachable and the request IS
 * well-formed - it was refused for a missing/wrong API key, which is a
 * client configuration problem (a service-account key, not a per-user
 * password), not a connectivity or model-availability problem.
 */
export function StatePanel({ kind, title, children }: { kind: StateKind; title?: string; children?: ReactNode }) {
  return (
    <div className={`state-panel state-${kind}`} role={kind === 'loading' ? 'status' : 'alert'}>
      <span className="glyph" aria-hidden="true">
        {GLYPH[kind]}
      </span>
      <div>
        <span className="state-title">{title ?? TITLE[kind]}</span>
        {children}
      </div>
    </div>
  )
}
