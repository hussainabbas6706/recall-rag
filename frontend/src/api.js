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

export async function getVault({ signal } = {}) {
  const response = await fetch(`${API_BASE_URL}/vault`, { signal })
  let data
  try {
    data = await response.json()
  } catch {
    throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`)
  }

  if (!response.ok) {
    throw new Error(data?.detail || `Vault listing failed with HTTP ${response.status}.`)
  }
  if (!Array.isArray(data.entries)) {
    throw new Error('The API returned an unexpected vault listing.')
  }
  return data.entries
}

export async function getVaultSelection({ signal } = {}) {
  const response = await fetch(`${API_BASE_URL}/vault/selection`, { signal })
  if (response.status === 204) {
    return null
  }
  const data = await readJsonResponse(response)
  if (!response.ok) {
    throw new Error(data?.detail || `Vault status failed with HTTP ${response.status}.`)
  }
  if (
    typeof data.vault_id !== 'string'
    || typeof data.name !== 'string'
    || !Array.isArray(data.entries)
  ) {
    throw new Error('The API returned an unexpected vault selection.')
  }
  return data
}

export async function selectVaultFolder() {
  const response = await fetch(`${API_BASE_URL}/vault/select`, { method: 'POST' })
  if (response.status === 204) {
    return null
  }
  const data = await readJsonResponse(response)
  if (!response.ok) {
    throw new Error(data?.detail || `Folder selection failed with HTTP ${response.status}.`)
  }
  return data
}

async function readJsonResponse(response) {
  try {
    return await response.json()
  } catch {
    throw new Error(`The API returned an unreadable response (HTTP ${response.status}).`)
  }
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

export function ingestVault(vaultId) {
  return postJson('/ingest', { vault_id: vaultId })
}

export function chatWithRecall(vaultId, query, history) {
  return postJson('/chat', { vault_id: vaultId, query, history })
}
