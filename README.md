# Recall

Search and chat with a personal Obsidian knowledge vault, with grounded answers and note citations.

## Project status

Phases 1–4 are implemented:

- **Phase 1 — Foundation:** API health check, Markdown-vault discovery, fixture vault, and local Qdrant configuration.
- **Phase 2 — Chunking:** Heading-aware Markdown chunks with source paths and heading metadata.
- **Phase 3 — Indexing and retrieval:** Gemini embeddings, Qdrant indexing, and semantic `/search`.
- **Phase 4 — Answer generation:** `/chat` retrieves context, generates a grounded answer, and returns numbered source/heading citation metadata. If retrieval finds no chunks, it returns `I don't have enough information to answer that.` with no citations.

**Phase 5 — Evaluation baseline and reliability checks:** offline failure-path tests, fixture-backed end-to-end coverage, deterministic retrieval regression checks, and a fixture-only live smoke check are in place. The small fixture set is not a substitute for evaluation on a larger representative vault. Citation metadata identifies retrieved sources but does not guarantee answer correctness.

## Tests and retrieval evaluation

Run the offline test suite with `uv run pytest -q`. A fixture-only retrieval regression check can be run with:

```powershell
uv run pytest tests/test_retrieval_evaluation.py -q -s
```

The benchmark uses four example questions, expected fixture source paths, in-memory Qdrant, and deterministic token vectors. It reports Hit@3 and mean reciprocal rank (MRR) so changes to the retrieval plumbing are visible and repeatable. These synthetic vectors do **not** measure Gemini embedding quality or predict performance on a personal vault. Live provider checks are optional and should use the fixture vault only.

In one optional live run using Gemini embeddings and generation with in-memory Qdrant, the four fixture queries had Hit@3 **1.00** and MRR **1.00**. An answerable chat query returned a fixture citation, and an unsupported question with retrieved context received the instructed fallback answer. This is a small snapshot, not a general quality guarantee; the API still returns metadata for all retrieved chunks and does not independently validate answer claims or citation support.

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
