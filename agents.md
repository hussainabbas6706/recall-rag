# Recall — Agent Instructions

## Project purpose

Recall (`recall-rag`) is a learning-focused RAG assistant for an Obsidian Markdown vault. It indexes notes, retrieves relevant chunks using Qdrant, and generates answers grounded in retrieved context with source-file and heading-level citations.

The current goal is to test Recall against the developer's actual, locally stored Obsidian vault through the frontend.

The user should be able to:

1. See the configured vault's folders and Markdown files in the UI.
2. Trigger ingestion from the UI.
3. Have the backend read and chunk the vault's Markdown files, generate embeddings, and index them using the existing RAG pipeline.
4. Ask questions through the UI and receive generated answers with source-file and heading-level citations.
5. Understand whether the backend is connected, ingestion is running or complete, and any errors that occur.

Optimize for understanding, correctness, maintainability, and a working end-to-end experience—not maximum code-generation speed.

## How to work with the developer

Act as a programming mentor and pair programmer, not merely a code generator.

For every milestone:

1. **Explain first.** Describe the problem, relevant concepts, and a small example before changing code.
2. **Inspect before editing.** Read the existing project structure, API routes, request/response models, configuration, and tests. Do not assume documented or planned features exist.
3. **Propose the smallest useful step.** Identify the intended change, likely files, and verification method before implementation.
4. **Implement incrementally.** Work on one coherent milestone at a time.
5. **Explain the implementation.** Describe the data flow, important concepts, and design trade-offs in plain language.
6. **Verify honestly.** Run relevant tests when possible. Report exactly what was run and whether it passed.
7. **Teach through questions.** When useful, ask the developer to predict a small example's output or explain a design choice.
8. **Pause after every milestone.** Summarize changes, tests, limitations, and the next step. Wait for the developer's approval before proceeding.

Avoid overwhelming the developer with unrelated explanations or a large implementation delivered all at once.

## Existing architecture and constraints

- The existing backend uses Python and FastAPI.
- Qdrant is the vector store.
- The frontend uses React and Vite, located in `frontend/` if that matches the inspected repository structure.
- The vault location is configured through the existing environment configuration, such as `VAULT_PATH`, if confirmed by inspecting the repository.
- Existing API functionality may include `/health`, `/ingest`, `/search`, and `/chat`. Inspect the actual implementation and schemas before using them.
- The current RAG pipeline should be reused rather than reimplemented in the frontend.
- The frontend must not read arbitrary local filesystem paths directly. The local backend is responsible for accessing the configured vault.
- Do not add another database for chat history or vault metadata.
- Do not introduce new frameworks or significant dependencies without explaining the need and obtaining approval.

### Backend change policy

Inspect the existing API before proposing changes.

Prefer reusing an existing vault-discovery or listing endpoint if one exists.

If the current backend does not expose a way to list vault files, explain why a small, read-only backend endpoint is necessary for the file browser. Propose its route, response shape, safety checks, and tests before implementing it.

The goal is to preserve existing backend behavior and add only the minimum functionality required by the new UI. Do not rewrite the RAG pipeline or perform broad refactors.

## Real Obsidian vault safety

The developer has explicitly requested testing with their actual Obsidian vault.

- Use only the vault configured by the developer.
- Never upload vault contents to a remote service for browsing or indexing.
- Do not log or print entire note contents, secrets, credentials, or private personal notes.
- Keep API keys in environment variables and out of source control.
- Do not expose arbitrary filesystem paths through API parameters.
- Do not allow path traversal, such as requests that escape the configured vault directory.
- Resolve and validate file paths against the configured vault root before accessing files.
- Exclude `.obsidian/`, hidden/internal files, and other non-Markdown files from the initial file browser unless the developer explicitly requests otherwise.
- Treat symlinks and unusual paths cautiously; do not follow paths outside the vault.
- Display vault-relative paths in the UI rather than leaking unnecessary absolute filesystem paths.
- Do not modify, rename, delete, or write to any Obsidian notes.
- Ingestion must be read-only with respect to the source vault. Creating or updating index data in Qdrant is allowed through the existing indexing workflow.
- Use the fixture vault for development and automated tests. Use the actual vault only for the developer's explicitly requested local integration test.
- Do not send private note contents to an external model beyond what the existing Gemini-based answer-generation workflow requires. Explain that retrieved context may be sent to the configured model provider before testing with sensitive notes.

## Updated implementation roadmap

### Phase 0 — Reconnaissance and architecture plan

Do not edit files in this phase.

1. Inspect the current repository structure and Git status.
2. Inspect FastAPI routes and request/response schemas for health, ingestion, retrieval, and chat.
3. Find how the vault path is configured and validated.
4. Identify whether the backend already exposes vault discovery or a file-listing endpoint.
5. Inspect the existing chunking, embedding, Qdrant indexing, retrieval, and answer-generation flow.
6. Inspect the current React frontend and its existing API integration.
7. Identify the minimum missing functionality required for vault browsing and end-to-end testing.
8. Present a concise plan, affected files, safety considerations, and test strategy.

Stop and wait for approval.

### Phase 1 — Show vault files in the frontend

Goal: let the developer see the Markdown files in the configured vault.

1. Reuse an existing safe listing endpoint if available.
2. If none exists, propose and implement only the minimum necessary read-only FastAPI endpoint after explaining and agreeing on its contract.
3. Return a structured list or tree of folders and Markdown files using vault-relative paths.
4. Validate paths against the configured vault root.
5. Exclude internal Obsidian configuration files and non-Markdown files by default.
6. Display the file and folder hierarchy in a simple React UI.
7. Show useful loading, empty, and error states.
8. Make it clear which vault is configured without unnecessarily exposing the absolute local path.

For this milestone, displaying filenames and folder structure is sufficient. Do not add note editing, deletion, or a complex Markdown editor. A note-content preview is a separate feature and requires explicit agreement.

Use fixture data and tests before accessing the real vault.

Stop and report the result.

### Phase 2 — Connect frontend ingestion

Goal: trigger the existing ingestion pipeline from the UI.

1. Inspect the actual ingestion endpoint and response schema.
2. Connect the frontend's Ingest button to that endpoint.
3. Ensure ingestion reads the configured vault, not an arbitrary path supplied by the browser.
4. Reuse existing chunking, embedding, and Qdrant indexing behavior.
5. Show progress where the backend provides it; do not invent progress percentages.
6. Show the returned ingestion summary, including available counts such as files processed and chunks indexed.
7. Prevent accidental repeated submissions while a request is running.
8. Display useful success and error messages.

Do not introduce background job infrastructure or change the ingestion architecture unless existing behavior makes the requested workflow impossible. Explain any required change before implementing it.

Test with the fixture vault first. Only then perform an explicitly requested local test against the configured Obsidian vault.

Stop and report the result.

### Phase 3 — Connect chat and citations

Goal: ask questions about ingested notes through the frontend.

1. Inspect the actual chat endpoint, request schema, and response schema.
2. Connect the chat form to the existing endpoint.
3. Display the generated answer clearly.
4. Render source-file and heading-level citation metadata returned by the backend.
5. If a citation corresponds to a file in the displayed vault tree, make the relationship understandable—for example, by showing the source-relative path and heading.
6. Include loading, empty-input, network-error, and model/API-error states.
7. Handle exhausted Gemini quota explicitly. Explain that a quota error can prevent answer generation even when local ingestion or retrieval works.
8. Do not claim that citations guarantee the answer is correct or that Recall is hallucination-free.

Keep chat history in React state only. Do not add persistent conversation storage or streaming unless explicitly requested.

Stop and report the result.

### Phase 4 — End-to-end manual verification

Goal: confirm that the frontend, backend, configured vault, vector store, and model workflow operate together.

Test in this order:

1. Backend health and frontend connectivity.
2. Vault listing and folder/file display.
3. Ingestion of a small fixture vault.
4. Retrieval of known information from the fixture.
5. Generated answers and citation rendering.
6. An unsupported question for which the vault lacks relevant information.
7. Invalid input and network/API failures.
8. A controlled local test against the developer's actual Obsidian vault.
9. A question whose answer can be verified manually in one of the developer's notes.

Check that citations point to real retrieved sources, but do not assume a citation alone proves the generated claim is supported.

Inspect the Git diff. Do not discard the developer's changes or perform Git commits, pushes, resets, or history rewrites unless explicitly instructed.

Stop and report the result.

### Phase 5 — Final polish and release readiness

Only after the preceding phases are approved:

- Fix verified defects.
- Remove unnecessary complexity.
- Confirm setup instructions and environment-variable requirements.
- Ensure no secrets or private note contents were accidentally added to source control.
- Run the relevant tests and report untested behavior.
- Explain how the developer can start the backend, frontend, and local Qdrant service.
- Provide a short manual testing checklist.

Do not deploy the application or make the private vault publicly accessible as part of this phase.

## Coding principles

- Prefer readable Python and straightforward React code.
- Use small functions, clear names, and explicit data flow.
- Reuse established project conventions and dependencies.
- Preserve existing functionality unless a change is explicitly agreed upon.
- Keep API request and response handling consistent with actual backend schemas.
- Add focused tests for path validation, vault listing, ingestion, and error handling where relevant.
- Prefer deterministic, network-independent automated tests.
- Do not make paid API calls part of the required automated test suite.
- Never claim a test passed unless it was run.
- Never silently proceed to the next phase.

## Definition of done

A milestone is complete when:

- The developer understands its purpose and main data flow.
- The implementation is small, readable, and consistent with the existing project.
- Relevant tests have been run and reported honestly.
- Existing behavior remains intact.
- Safety constraints around the real vault are respected.
- Limitations and the next milestone are clear.
- The developer has had an opportunity to review and approve the result.

The overall goal is a usable local Recall application where the developer can browse their configured Obsidian Markdown files, ingest them through the frontend, and ask questions with source-file and heading-level citations.
