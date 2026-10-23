import { useEffect, useState } from 'react'
import './App.css'
import { API_BASE_URL, chatWithRecall, getHealth, ingestVault } from './api.js'

function App() {
  const [health, setHealth] = useState({ status: 'checking', error: '' })
  const [healthCheckId, setHealthCheckId] = useState(0)
  const [ingestion, setIngestion] = useState({ status: 'idle', result: null, error: '' })
  const [question, setQuestion] = useState('')
  const [chat, setChat] = useState({ status: 'idle', question: '', result: null, error: '' })

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

  async function handleIngest() {
    setIngestion({ status: 'loading', result: null, error: '' })
    try {
      const result = await ingestVault()
      setIngestion({ status: 'success', result, error: '' })
    } catch (error) {
      setIngestion({ status: 'error', result: null, error: error.message })
    }
  }

  async function handleChatSubmit(event) {
    event.preventDefault()
    const submittedQuestion = question.trim()
    if (!submittedQuestion) {
      setChat({
        status: 'error',
        question: '',
        result: null,
        error: 'Enter a question before sending.',
      })
      return
    }

    setChat({ status: 'loading', question: submittedQuestion, result: null, error: '' })
    try {
      const result = await chatWithRecall(submittedQuestion)
      setChat({ status: 'success', question: submittedQuestion, result, error: '' })
    } catch (error) {
      setChat({
        status: 'error',
        question: submittedQuestion,
        result: null,
        error: error.message,
      })
    }
  }

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
          <p className="panel-note">
            The vault path is configured on the backend. This page cannot browse your computer’s folders.
          </p>
          <button
            className="button button-secondary"
            type="button"
            onClick={handleIngest}
            disabled={ingestion.status === 'loading'}
          >
            {ingestion.status === 'loading' ? 'Indexing vault…' : 'Ingest vault'}
          </button>
          {ingestion.status === 'error' && (
            <p className="feedback feedback-error" role="alert">{ingestion.error}</p>
          )}
          {ingestion.status === 'success' && (
            <div className="feedback feedback-success" role="status">
              <p>{ingestion.result.message}</p>
              <p>
                {ingestion.result.files_found} files · {ingestion.result.chunks_created} chunks
                created · {ingestion.result.chunks_indexed} indexed
              </p>
            </div>
          )}
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
          <form onSubmit={handleChatSubmit}>
            <label className="visually-hidden" htmlFor="question">
              Your question
            </label>
            <textarea
              id="question"
              className="question-input"
              placeholder="Ask a question about your notes…"
              rows="3"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={chat.status === 'loading'}
            />
            <div className="chat-actions">
              <span className="panel-note">Answers are generated from retrieved note chunks.</span>
              <button
                className="button button-primary"
                type="submit"
                disabled={chat.status === 'loading'}
              >
                {chat.status === 'loading' ? 'Thinking…' : 'Send'}
                {chat.status !== 'loading' && <span aria-hidden="true">→</span>}
              </button>
            </div>
          </form>
          {chat.status === 'error' && (
            <p className="feedback feedback-error" role="alert">{chat.error}</p>
          )}
          {chat.status === 'loading' && (
            <p className="feedback" role="status">Searching your notes and generating an answer…</p>
          )}
          {chat.status === 'success' && chat.result && (
            <section className="chat-result" aria-label="Answer">
              <div className="submitted-question">
                <p className="eyebrow">YOUR QUESTION</p>
                <p>{chat.question}</p>
              </div>
              <div className="answer-block">
                <p className="eyebrow">RECALL’S ANSWER</p>
                <p className="answer-text">{chat.result.answer}</p>
              </div>
              <div className="citations">
                <h3>Sources</h3>
                {Array.isArray(chat.result.citations) && chat.result.citations.length > 0 ? (
                  <ol className="citation-list">
                    {chat.result.citations.map((citation) => (
                      <li key={`${citation.reference}-${citation.source_path}-${citation.heading_path.join('/')}`}>
                        <span className="citation-reference">[{citation.reference}]</span>
                        <span className="citation-source">
                          <span>{citation.source_path}</span>
                          <span className="citation-heading">
                            {citation.heading_path.length > 0
                              ? citation.heading_path.join(' › ')
                              : '(root section)'}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="panel-note">
                    No source citations were returned for this answer.
                  </p>
                )}
              </div>
            </section>
          )}
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
