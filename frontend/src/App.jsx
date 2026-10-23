import { useEffect, useState } from 'react'
import './App.css'
import { API_BASE_URL, getHealth } from './api.js'

function App() {
  const [health, setHealth] = useState({ status: 'checking', error: '' })
  const [healthCheckId, setHealthCheckId] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setHealth({ status: 'checking', error: '' })

    getHealth({ signal: controller.signal })
      .then((result) => {
        setHealth({ status: 'connected', service: result.service, error: '' })
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setHealth({ status: 'unavailable', error: error.message })
        }
      })

    return () => controller.abort()
  }, [healthCheckId])

  const statusMessage = {
    checking: 'Checking Recall API…',
    connected: `Connected to ${health.service || 'Recall API'}`,
    unavailable: 'Recall API is unavailable',
  }[health.status]

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Recall home">
          <span className="brand-mark" aria-hidden="true">R</span>
          <span>recall</span>
        </a>
        <span className="local-label">
          <span className="local-dot" aria-hidden="true" />
          Local workspace
        </span>
      </header>

      <section className="welcome" aria-labelledby="welcome-title">
        <p className="eyebrow">YOUR PERSONAL KNOWLEDGE ASSISTANT</p>
        <h1 id="welcome-title">Bring your notes back to mind.</h1>
        <p className="welcome-copy">
          Search your Obsidian knowledge with answers grounded in your notes.
        </p>
      </section>

      <section className="workspace" aria-label="Recall workspace">
        <article className="panel backend-panel" aria-labelledby="backend-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">01</div>
            <div>
              <p className="eyebrow">CONNECTION</p>
              <h2 id="backend-title">Backend status</h2>
            </div>
          </div>
          <div
            className={`status-row status-${health.status}`}
            role="status"
            aria-live="polite"
          >
            <span className="status-indicator" aria-hidden="true" />
            <span>{statusMessage}</span>
          </div>
          {health.status === 'unavailable' && (
            <p className="health-error">{health.error || 'Check that the backend is running.'}</p>
          )}
          <p className="panel-note">Recall API · <code>{API_BASE_URL}</code></p>
          <button
            className="button button-secondary health-retry"
            type="button"
            onClick={() => setHealthCheckId((current) => current + 1)}
            disabled={health.status === 'checking'}
          >
            {health.status === 'checking' ? 'Checking…' : 'Check again'}
          </button>
        </article>

        <article className="panel ingest-panel" aria-labelledby="ingest-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">02</div>
            <div>
              <p className="eyebrow">YOUR KNOWLEDGE</p>
              <h2 id="ingest-title">Vault ingestion</h2>
            </div>
          </div>
          <p className="panel-copy">
            Index the Markdown notes from the vault configured by the backend.
          </p>
          <button className="button button-secondary" type="button" disabled>
            Ingest vault
          </button>
          <p className="panel-note">Ingestion controls are coming in a later phase.</p>
        </article>

        <article className="panel chat-panel" aria-labelledby="chat-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">03</div>
            <div>
              <p className="eyebrow">ASK YOUR NOTES</p>
              <h2 id="chat-title">Chat with Recall</h2>
            </div>
          </div>
          <p className="panel-copy">
            Ask a question and get an answer with links back to its source notes.
          </p>
          <label className="visually-hidden" htmlFor="question">
            Your question
          </label>
          <textarea
            id="question"
            className="question-input"
            placeholder="Your question will go here…"
            rows="3"
            disabled
          />
          <div className="chat-actions">
            <span className="panel-note">Chat will be available in a later phase.</span>
            <button className="button button-primary" type="button" disabled>
              Send <span aria-hidden="true">→</span>
            </button>
          </div>
        </article>
      </section>

      <footer className="footer">
        <span>Recall v1.1</span>
        <span>Private by design · Runs locally</span>
      </footer>
    </main>
  )
}

export default App
