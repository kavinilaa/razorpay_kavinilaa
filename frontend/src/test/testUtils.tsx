import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'

import { ModelInfoProvider } from '../context/ModelInfoContext'

/** Wraps a page/component under test with the same providers App.tsx uses
 * (router + model-info context), so pages that call useModelInfo() (e.g. for
 * the operating-mode metrics or the persistent banner) render without
 * throwing. Tests must mock '../api' (see individual test files) so this
 * never makes a real network call. */
export function renderWithProviders(ui: ReactElement) {
  return render(
    <MemoryRouter>
      <ModelInfoProvider>{ui}</ModelInfoProvider>
    </MemoryRouter>,
  )
}
