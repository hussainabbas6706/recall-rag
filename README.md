# Recall

Search and chat with a personal Obsidian knowledge vault, with grounded answers and note citations.

## Project status

Phases 1–4 are implemented:

- **Phase 1 — Foundation:** API health check, Markdown-vault discovery, fixture vault, and local Qdrant configuration.
- **Phase 2 — Chunking:** Heading-aware Markdown chunks with source paths and heading metadata.
- **Phase 3 — Indexing and retrieval:** Gemini embeddings, Qdrant indexing, and semantic `/search`.
- **Phase 4 — Answer generation:** `/chat` retrieves context, generates a grounded answer, and returns numbered source/heading citation metadata. If retrieval finds no chunks, it returns `I don't have enough information to answer that.` with no citations.

**Phase 5 — Evaluation and polish remains:** evaluate retrieval and citation quality, unsupported questions, error handling, and end-to-end behavior. Live provider behavior should be validated with the fixture vault; citations are source metadata and do not guarantee answer correctness.

## Local setup

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to use Gemini embeddings and answer generation.
2. Install dependencies: `uv sync --all-groups`.
3. Start Qdrant: `docker compose up -d`.
4. Run the API: `uv run uvicorn app.main:app --reload`.
5. Visit `http://127.0.0.1:8000/docs`, call `POST /ingest` to index the fixture vault, then use `POST /search` or `POST /chat`.

Inspect the fixture vault from the terminal with:

```powershell
uv run python -m app.cli inspect
```

## Safety

`vault/`, `qdrant_storage/`, and `.env` are ignored by Git. Configure `VAULT_PATH` to point to a local vault; do not place a personal vault in this repository.
