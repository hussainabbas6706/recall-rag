import { useEffect, useState } from 'react'
import './App.css'
import {
  API_BASE_URL,
  chatWithRecall,
  getHealth,
  getVaultSelection,
  ingestVault,
  selectVaultFolder,
} from './api.js'

function buildVaultTree(entries) {
  const root = { children: [] }
  for (const entry of entries) {
    let parent = root
    const parts = entry.path.split('/')
    parts.forEach((name, index) => {
      const path = parts.slice(0, index + 1).join('/')
      let node = parent.children.find((child) => child.path === path)
      if (!node) {
        node = {
          name,
          path,
          kind: index === parts.length - 1 ? entry.kind : 'folder',
          children: [],
        }
        parent.children.push(node)
      }
      parent = node
    })
  }
  return root.children
}

function VaultTreeNodes({ nodes }) {
  return (
    <ul className="vault-tree">
      {nodes.map((node) => (
        <li key={node.path}>
          {node.kind === 'folder' ? (
            <details open>
              <summary>{node.name}</summary>
              {node.children.length > 0 && <VaultTreeNodes nodes={node.children} />}
            </details>
          ) : (
            <span className="vault-file" id={`vault-file-${encodeURIComponent(node.path)}`}>
              {node.name}
            </span>
          )}
        </li>
      ))}
    </ul>
  )
}

function ChatTurn({ message }) {
  const isUser = message.role === 'user'
  return (
    <article className={`chat-turn ${isUser ? 'chat-turn-user' : 'chat-turn-assistant'}`}>
      <p className="eyebrow">{isUser ? 'YOU' : 'RECALL'}</p>
      <p className={isUser ? 'chat-question-text' : 'answer-text'}>{message.content}</p>
      {!isUser && (
        <div className="citations">
          <h3>Sources</h3>
          {message.citations.length > 0 ? (
            <ol className="citation-list">
              {message.citations.map((citation) => (
                <li key={`${citation.reference}-${citation.source_path}-${citation.heading_path.join('/')}`}>
                  <span className="citation-reference">[{citation.reference}]</span>
                  <a
                    className="citation-link"
                    href={`#vault-file-${encodeURIComponent(citation.source_path)}`}
                  >
                    <span>{citation.source_path}</span>
                    <span className="citation-heading">
                      {citation.heading_path.length > 0
                        ? citation.heading_path.join(' › ')
                        : '(root section)'}
                    </span>
                  </a>
                </li>
              ))}
            </ol>
          ) : (
            <p className="panel-note">No source citations were returned for this answer.</p>
          )}
        </div>
      )}
    </article>
  )
}

function App() {
  const [health, setHealth] = useState({ status: 'checking', error: '' })
  const [healthCheckId, setHealthCheckId] = useState(0)
  const [vault, setVault] = useState({ status: 'checking', selection: null, error: '' })
  const [selectingVault, setSelectingVault] = useState(false)
  const [selectionError, setSelectionError] = useState('')
  const [ingestion, setIngestion] = useState({ status: 'idle', vaultId: null, result: null, error: '' })
  const [question, setQuestion] = useState('')
  const [chatStatus, setChatStatus] = useState('idle')
  const [chatError, setChatError] = useState('')
  const [messages, setMessages] = useState([])

  useEffect(() => {
    const controller = new AbortController()
    setHealth({ status: 'checking', error: '' })
    getHealth({ signal: controller.signal })
      .then((result) => setHealth({ status: 'connected', service: result.service, error: '' }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setHealth({ status: 'unavailable', error: error.message })
        }
      })
    return () => controller.abort()
  }, [healthCheckId])

  useEffect(() => {
    const controller = new AbortController()
    setVault({ status: 'checking', selection: null, error: '' })
    getVaultSelection({ signal: controller.signal })
      .then((selection) => {
        setVault({ status: selection ? 'connected' : 'empty', selection, error: '' })
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setVault({ status: 'error', selection: null, error: error.message })
        }
      })
    return () => controller.abort()
  }, [])

  const statusMessage = {
    checking: 'Checking Recall API…',
    connected: `Connected to ${health.service || 'Recall API'}`,
    unavailable: 'Recall API is unavailable',
  }[health.status]
  const selectedVault = vault.selection
  const isIndexed = ingestion.status === 'success' && ingestion.vaultId === selectedVault?.vault_id
  const busy = selectingVault || ingestion.status === 'loading' || chatStatus === 'loading'

  async function handleSelectVault() {
    setSelectingVault(true)
    setSelectionError('')
    try {
      const selection = await selectVaultFolder()
      if (!selection) {
        return
      }
      setVault({ status: 'connected', selection, error: '' })
      setIngestion({ status: 'idle', vaultId: null, result: null, error: '' })
      setMessages([])
      setQuestion('')
      setChatStatus('idle')
      setChatError('')
    } catch (error) {
      setSelectionError(error.message)
    } finally {
      setSelectingVault(false)
    }
  }

  async function handleIngest() {
    if (!selectedVault) {
      return
    }
    setIngestion({ status: 'loading', vaultId: selectedVault.vault_id, result: null, error: '' })
    try {
      const result = await ingestVault(selectedVault.vault_id)
      setIngestion({ status: 'success', vaultId: selectedVault.vault_id, result, error: '' })
      setMessages([])
      setChatStatus('idle')
      setChatError('')
    } catch (error) {
      setIngestion({
        status: 'error',
        vaultId: selectedVault.vault_id,
        result: null,
        error: error.message,
      })
    }
  }

  async function handleChatSubmit(event) {
    event.preventDefault()
    const submittedQuestion = question.trim()
    if (!submittedQuestion) {
      setChatStatus('error')
      setChatError('Enter a question before sending.')
      return
    }
    if (!selectedVault || !isIndexed) {
      setChatStatus('error')
      setChatError('Select and index a vault before asking questions.')
      return
    }

    setChatStatus('loading')
    setChatError('')
    const history = messages.map(({ role, content }) => ({ role, content }))
    try {
      const result = await chatWithRecall(selectedVault.vault_id, submittedQuestion, history)
      setMessages((current) => [
        ...current,
        { role: 'user', content: submittedQuestion },
        { role: 'assistant', content: result.answer, citations: result.citations || [] },
      ])
      setQuestion('')
      setChatStatus('idle')
    } catch (error) {
      setChatStatus('error')
      setChatError(error.message)
    }
  }

  const tree = selectedVault ? buildVaultTree(selectedVault.entries) : []

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
          Connect a vault folder, index its notes, and chat with answers grounded in your sources.
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
          <div className={`status-row status-${health.status}`} role="status" aria-live="polite">
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

        <article className="panel ingest-panel" aria-labelledby="vault-connect-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">02</div>
            <div>
              <p className="eyebrow">LOCAL VAULT</p>
              <h2 id="vault-connect-title">Connect a vault</h2>
            </div>
          </div>
          <p className="panel-copy">
            Choose a folder using the operating system’s folder dialog. Recall reads Markdown
            files from that folder; the source notes are never modified.
          </p>
          <button
            className="button button-secondary"
            type="button"
            onClick={handleSelectVault}
            disabled={busy || health.status !== 'connected'}
          >
            {selectingVault ? 'Choose a folder in the system dialog…' : 'Select Vault Folder'}
          </button>
          {selectedVault && (
            <div className="vault-connection" role="status">
              <span className="status-indicator" aria-hidden="true" />
              <span><strong>{selectedVault.name}</strong> · {isIndexed ? 'Indexed' : 'Connected — not indexed'}</span>
            </div>
          )}
          {vault.status === 'checking' && (
            <p className="feedback" role="status">Checking for an active vault selection…</p>
          )}
          {selectionError && <p className="feedback feedback-error" role="alert">{selectionError}</p>}
          {vault.status === 'error' && (
            <p className="feedback feedback-error" role="alert">{vault.error}</p>
          )}
          <p className="panel-note">
            Switching vaults starts a fresh isolated index; the previous Qdrant index is retained
            but is no longer used for this chat.
          </p>
        </article>

        <article className="panel ingest-panel" aria-labelledby="ingest-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">03</div>
            <div>
              <p className="eyebrow">INDEX SELECTED NOTES</p>
              <h2 id="ingest-title">Vault ingestion</h2>
            </div>
          </div>
          <p className="panel-copy">
            Index Markdown files from {selectedVault ? selectedVault.name : 'the selected vault'}.
            Chunk embeddings run locally on the CPU; the first run may download the model.
          </p>
          <button
            className="button button-secondary"
            type="button"
            onClick={handleIngest}
            disabled={!selectedVault || busy}
          >
            {ingestion.status === 'loading' ? 'Indexing vault…' : 'Ingest vault'}
          </button>
          {ingestion.status === 'error' && (
            <p className="feedback feedback-error" role="alert">{ingestion.error}</p>
          )}
          {ingestion.status === 'success' && (
            <div
              className={`feedback ${
                ingestion.result.status === 'partial' ? 'feedback-warning' : 'feedback-success'
              }`}
              role={ingestion.result.status === 'partial' ? 'alert' : 'status'}
            >
              <p>{ingestion.result.message}</p>
              <p>
                {ingestion.result.files_found} files discovered · {ingestion.result.files_indexed} new ·{' '}
                {ingestion.result.files_updated} updated · {ingestion.result.files_unchanged} unchanged ·{' '}
                {ingestion.result.files_deleted} deleted · {ingestion.result.chunks_indexed} chunks indexed
              </p>
              {ingestion.result.failures?.length > 0 && (
                <ul className="ingestion-failures">
                  {ingestion.result.failures.map((failure) => (
                    <li key={`${failure.source_path}-${failure.reason}`}>
                      <strong>{failure.source_path}:</strong> {failure.reason}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </article>

        <article className="panel vault-panel" aria-labelledby="vault-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">04</div>
            <div>
              <p className="eyebrow">SELECTED VAULT</p>
              <h2 id="vault-title">{selectedVault ? selectedVault.name : 'Vault files'}</h2>
            </div>
          </div>
          <p className="panel-copy">
            {selectedVault
              ? 'Folders and Markdown filenames only. Note contents are not returned by the listing.'
              : 'Select a folder to view its Markdown files.'}
          </p>
          {selectedVault && tree.length === 0 && (
            <p className="feedback" role="status">No folders or Markdown files were found.</p>
          )}
          {tree.length > 0 && (
            <div className="vault-tree-container" aria-label="Selected vault file tree">
              <VaultTreeNodes nodes={tree} />
            </div>
          )}
        </article>

        <article className="panel chat-panel" aria-labelledby="chat-title">
          <div className="panel-heading">
            <div className="section-icon" aria-hidden="true">05</div>
            <div>
              <p className="eyebrow">GROUNDED CONVERSATION</p>
              <h2 id="chat-title">Chat with {selectedVault?.name || 'your vault'}</h2>
            </div>
          </div>
          <p className="panel-copy">
            Follow-up questions use this conversation and retrieve evidence from the selected
            vault’s isolated index.
          </p>
          <div className="chat-transcript" aria-live="polite">
            {messages.map((message, index) => (
              <ChatTurn key={`${message.role}-${index}`} message={message} />
            ))}
            {messages.length === 0 && (
              <p className="panel-note">
                {isIndexed
                  ? 'Ask a question to begin.'
                  : 'Select and ingest a vault before asking a question.'}
              </p>
            )}
            {chatStatus === 'loading' && (
              <p className="feedback" role="status">Searching this vault and generating an answer…</p>
            )}
          </div>
          <form className="chat-form" onSubmit={handleChatSubmit}>
            <label className="visually-hidden" htmlFor="question">Your question</label>
            <textarea
              id="question"
              className="question-input"
              placeholder={isIndexed ? 'Ask a question or follow up…' : 'Select and ingest a vault first'}
              rows="3"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={!isIndexed || busy}
            />
            <div className="chat-actions">
              <span className="panel-note">Answers may send retrieved note context to Gemini.</span>
              <button
                className="button button-primary"
                type="submit"
                disabled={!isIndexed || busy}
              >
                {chatStatus === 'loading' ? 'Thinking…' : 'Send'}
                {chatStatus !== 'loading' && <span aria-hidden="true">→</span>}
              </button>
            </div>
          </form>
          {chatStatus === 'error' && (
            <p className="feedback feedback-error" role="alert">{chatError}</p>
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
