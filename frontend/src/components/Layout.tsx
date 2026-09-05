import { NavLink, Outlet } from 'react-router-dom'

export function Layout() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-title">AI Fraud-Spike &amp; Risk Detection</span>
      </header>
      <div className="app-workspace">
        <aside className="app-sidebar" aria-label="Merchant review tools">
          <nav className="sidebar-nav">
            <NavLink to="/" end>
              Transaction Scoring
            </NavLink>
            <NavLink to="/spikes">Spike Analysis</NavLink>
          </nav>
        </aside>
        <main className="app-main">
          <Outlet />
        </main>
      </div>
      <footer className="app-footer">
        AI Risk Manager prototype · Detection and review support only
      </footer>
    </div>
  )
}
