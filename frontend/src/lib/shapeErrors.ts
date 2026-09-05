import type { PydanticValidationError } from '../api'

/**
 * Groups FastAPI's automatic shape-validation errors by request-body field
 * name (the last segment of `loc` when `loc[0] === "body"`), so a form can
 * render each error under its own field instead of as a generic toast - a
 * cross-cutting requirement of this phase.
 */
export function groupShapeErrorsByField(errors: PydanticValidationError[]): {
  byField: Record<string, string[]>
  general: string[]
} {
  const byField: Record<string, string[]> = {}
  const general: string[] = []

  for (const err of errors) {
    if (err.loc[0] === 'body' && typeof err.loc[err.loc.length - 1] === 'string') {
      const field = err.loc[err.loc.length - 1] as string
      byField[field] = [...(byField[field] ?? []), err.msg]
    } else {
      general.push(`${err.loc.join('.')}: ${err.msg}`)
    }
  }

  return { byField, general }
}
