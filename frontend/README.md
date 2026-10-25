# Recall frontend

Minimal React + Vite shell for the local Recall API.

## Run locally

From this directory:

```powershell
npm install
npm run dev
```

The development server prints its local URL (usually `http://localhost:5173`).
The browser calls the same-origin `/api` path; Vite proxies those requests to
`http://127.0.0.1:8000` by default. To use a different backend in development,
set `RECALL_API_PROXY_TARGET` before starting Vite. Never put API keys or
secrets in frontend environment variables.

The frontend checks `GET /health`, can ingest the backend-configured vault
with `POST /ingest`, and can submit questions to `POST /chat`. The browser
also displays vault-relative folder and Markdown filenames from the read-only
`GET /vault` endpoint. The listing does not include note contents or the
backend's absolute vault path. The browser does not select the backend's
local vault.
