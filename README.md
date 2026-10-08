# Recall

Search and chat with a personal Obsidian knowledge vault, with grounded answers and note citations.

## Phase 1 status

The project currently has an API health check, Markdown-vault discovery, a fixture vault, and local Qdrant configuration. Heading-aware chunking, embeddings, retrieval, and answer generation will be implemented in subsequent phases.

## Local setup

1. Copy `.env.example` to `.env`, then set `GEMINI_API_KEY` when we reach the real embedding validation.
2. Install dependencies: `uv sync --all-groups`.
3. Start Qdrant: `docker compose up -d`.
4. Run the API: `uv run uvicorn app.main:app --reload`.
5. Visit `http://127.0.0.1:8000/docs` and call `POST /ingest`.

Inspect the fixture vault from the terminal with:

```powershell
uv run python -m app.cli inspect
```

## Safety

`vault/`, `qdrant_storage/`, and `.env` are ignored by Git. Configure `VAULT_PATH` to point to a local vault; do not place a personal vault in this repository.
