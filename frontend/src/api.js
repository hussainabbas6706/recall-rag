export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || '/api'

export async function getHealth({ signal } = {}) {
  const response = await fetch(`${API_BASE_URL}/health`, { signal })

  if (!response.ok) {
    throw new Error(`Health check failed with HTTP ${response.status}.`)
  }

  const health = await response.json()
  if (health.status !== 'ok' || typeof health.service !== 'string') {
    throw new Error('The API returned an unexpected health response.')
  }

  return health
}

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  let data
  try {
    data = await response.json()
  } catch {
    throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`)
  }

  if (!response.ok) {
    const detail = data?.detail
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).filter(Boolean).join('; ')
      : detail
    throw new Error(message || `Request failed with HTTP ${response.status}.`)
  }

  return data
}

export function ingestVault() {
  return postJson('/ingest', {})
}

export function chatWithRecall(query) {
  return postJson('/chat', { query })
}
