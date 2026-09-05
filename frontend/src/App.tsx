import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { Layout } from './components/Layout'
import { SpikeAnalysisPage } from './pages/SpikeAnalysisPage'
import { TransactionScoringPage } from './pages/TransactionScoringPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<TransactionScoringPage />} />
          <Route path="spikes" element={<SpikeAnalysisPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
