# Recall

Search and chat with a personal Obsidian knowledge vault, with grounded answers and note citations.

## Project status

Phases 1–4 are implemented:

- **Phase 1 — Foundation:** API health check, Markdown-vault discovery, fixture vault, and local Qdrant configuration.
- **Phase 2 — Chunking:** Heading-aware Markdown chunks with source paths and heading metadata.
- **Phase 3 — Indexing and retrieval:** local Sentence Transformers embeddings, incremental Qdrant indexing, and semantic `/search`.
- **Phase 4 — Answer generation:** `/chat` retrieves context, generates a grounded answer, and returns numbered source/heading citation metadata. If retrieval finds no chunks, it returns `I don't have enough information to answer that.` with no citations.

**Phase 5 — Evaluation baseline and reliability checks:** offline failure-path tests, fixture-backed end-to-end coverage, deterministic retrieval regression checks, and a fixture-only live smoke check are in place. The small fixture set is not a substitute for evaluation on a larger representative vault. Citation metadata identifies retrieved sources but does not guarantee answer correctness.

## Tests and retrieval evaluation

Run the offline test suite with `uv run pytest -q`. A fixture-only retrieval regression check can be run with:

```powershell
uv run pytest tests/test_retrieval_evaluation.py -q -s
```

The benchmark uses four example questions, expected fixture source paths, in-memory Qdrant, and deterministic token vectors. It reports Hit@3 and mean reciprocal rank (MRR) so changes to the retrieval plumbing are visible and repeatable. These synthetic vectors do **not** measure the MiniLM model's retrieval quality or predict performance on a personal vault.

The deterministic benchmark is not a quality guarantee; the API returns metadata for retrieved chunks but does not independently validate answer claims or citation support.

## Local setup

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` for Gemini answer generation. Note embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2` on CPU; its first use downloads the model if it is not cached.
2. Install dependencies: `uv sync --all-groups`.
3. Start Qdrant: `docker compose up -d`.
4. Run the API: `uv run uvicorn app.main:app --reload`.
5. Start the frontend with `npm.cmd --prefix frontend run dev`, open the Vite URL, select a fixture or local vault folder, and click **Ingest vault**. Ask questions from the UI; chat generation continues to require Gemini API connectivity and credentials.

Each selected vault and embedding-model version uses a stable, separate Qdrant collection. Re-ingestion compares note content hashes, skips unchanged notes, replaces changed notes, and removes vectors for deleted notes. Existing Gemini-embedding collections are left untouched. Gemini is still used for answer generation, so retrieved note context is sent to Google when a chat answer is requested.

Inspect the fixture vault from the terminal with:

```powershell
uv run python -m app.cli inspect
```

## Safety

`vault/`, `qdrant_storage/`, and `.env` are ignored by Git. Configure `VAULT_PATH` to point to a local vault; do not place a personal vault in this repository.
