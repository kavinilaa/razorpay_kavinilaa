import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'

import { fetchModelInfo } from '../api'
import type { ApiOutcome } from '../api'
import type { ModelInfoResponse } from '../api'

/** A default, hard-coded fallback caveat so the persistent banner NEVER goes
 * blank, even if GET /model-info hasn't resolved yet or failed outright
 * (network error). The wording matches model_info.py's own literal
 * `data_provenance` field so it's never contradicted once the real fetch
 * completes. */
export const FALLBACK_SYNTHETIC_DATA_CAVEAT =
  'This is a prototype trained and evaluated only on the synthetic PaySim dataset. It has not been validated on real transaction data.'

interface ModelInfoContextValue {
  outcome: ApiOutcome<ModelInfoResponse> | { kind: 'loading' }
  /** Always resolvable: the real fetched caveat once available, otherwise the hard-coded fallback. */
  syntheticDataCaveat: string
  refetch: () => void
}

const ModelInfoContext = createContext<ModelInfoContextValue | undefined>(undefined)

export function ModelInfoProvider({ children }: { children: ReactNode }) {
  const [outcome, setOutcome] = useState<ApiOutcome<ModelInfoResponse> | { kind: 'loading' }>({
    kind: 'loading',
  })
  const [refetchCounter, setRefetchCounter] = useState(0)

  useEffect(() => {
    let cancelled = false
    setOutcome({ kind: 'loading' })
    fetchModelInfo().then((result) => {
      if (!cancelled) setOutcome(result)
    })
    return () => {
      cancelled = true
    }
  }, [refetchCounter])

  const syntheticDataCaveat =
    outcome.kind === 'success' ? outcome.data.data_provenance : FALLBACK_SYNTHETIC_DATA_CAVEAT

  return (
    <ModelInfoContext.Provider
      value={{ outcome, syntheticDataCaveat, refetch: () => setRefetchCounter((c) => c + 1) }}
    >
      {children}
    </ModelInfoContext.Provider>
  )
}

export function useModelInfo(): ModelInfoContextValue {
  const ctx = useContext(ModelInfoContext)
  if (!ctx) throw new Error('useModelInfo must be used within a ModelInfoProvider')
  return ctx
}
