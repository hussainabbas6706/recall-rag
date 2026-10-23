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
