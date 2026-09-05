import { describe, expect, it } from 'vitest'

import type { PydanticValidationError } from '../api'
import { groupShapeErrorsByField } from './shapeErrors'

describe('groupShapeErrorsByField', () => {
  it('groups body-field errors under their field name', () => {
    const errors: PydanticValidationError[] = [
      { type: 'greater_than_equal', loc: ['body', 'amount'], msg: 'Input should be >= 0' },
      { type: 'missing', loc: ['body', 'step'], msg: 'Field required' },
    ]
    const { byField, general } = groupShapeErrorsByField(errors)
    expect(byField.amount).toEqual(['Input should be >= 0'])
    expect(byField.step).toEqual(['Field required'])
    expect(general).toEqual([])
  })

  it('collects non-body errors into general', () => {
    const errors: PydanticValidationError[] = [
      { type: 'enum', loc: ['query', 'operating_mode'], msg: 'Input should be a valid enumeration member' },
    ]
    const { byField, general } = groupShapeErrorsByField(errors)
    expect(byField).toEqual({})
    expect(general).toHaveLength(1)
    expect(general[0]).toContain('operating_mode')
  })

  it('accumulates multiple errors for the same field', () => {
    const errors: PydanticValidationError[] = [
      { type: 'a', loc: ['body', 'amount'], msg: 'first' },
      { type: 'b', loc: ['body', 'amount'], msg: 'second' },
    ]
    const { byField } = groupShapeErrorsByField(errors)
    expect(byField.amount).toEqual(['first', 'second'])
  })
})
