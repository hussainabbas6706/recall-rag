# Recall v1.1 Frontend --- Phased Agent Instructions

## Project goal
Build a simple, usable React + Vite frontend for the existing Recall RAG
project. The purpose is to make the completed local naive RAG MVP easy
to use through a browser.

Work **one phase at a time**. Do not implement all phases in one prompt
or one pass. At the end of each phase, summarize the changes, explain
how to test them, and wait for the user to approve moving to the next
phase.

## Hard constraints

- Frontend only. **Do not modify backend files**: no FastAPI routes,
request/response schemas, RAG logic, ingestion logic, embeddings,
Qdrant configuration, backend dependencies, or backend tests.
- Do not invent endpoints, payload fields, response fields, or
behavior. Inspect the existing backend code and README to learn the
actual API contract, but do not edit the backend.
- Keep Recall v1 complete and unchanged.
- Put the UI in `frontend/` unless the repository already has a
frontend directory or an existing convention clearly says otherwise.
- Use React + Vite, plain CSS, and native `fetch`.
- Avoid unnecessary dependencies, UI component libraries, routing,
global state libraries, animations, and elaborate design systems.
- No authentication, deployment, cloud functionality, streaming,
persistent chat history, agent features, or advanced RAG
functionality.
- Do not expose API keys or secrets in frontend code or `VITE_`
environment variables.
- Target a small local-first UI. Prefer correctness and usability over
decorative polish.

## Required workflow for every phase

1. Read the relevant existing files before making changes.
2. State the intended changes briefly.
3. Implement only the current phase.
4. Inspect the diff and ensure no backend files changed.
5. Run the relevant available checks.
6. Report files changed, commands/tests run, results, and any blockers.
7. Stop and wait for approval before starting the next phase.
Do not skip ahead, silently start the next phase, or combine phases
without explicit user approval.

---

## Phase 0 --- Repository reconnaissance and API contract
**Goal:** Understand the existing project before writing frontend code.

Tasks:
- Inspect the repository tree, README, backend route handlers,
request/response schemas, and relevant configuration.
- Confirm the actual behavior and formats of `GET /health`, `POST /ingest`, and
`POST /chat`.
- Inspect `POST /search` only if it is relevant to the existing UI scope.
- Confirm how the vault path is configured, including any `VAULT_PATH` environment variable.
- Identify the backend base URL/port and existing CORS behavior.
- Report the actual API request and response shapes, with source file paths.
- Identify any UI limitation that cannot be solved without backend changes.

Restrictions:
- Do not edit files during this phase.
- Do not assume a folder-picker endpoint exists.
- Do not claim that the browser can freely access an arbitrary local folder.

**Exit criteria:** A concise, code-grounded API contract and
implementation outline are presented and approved by the user.

## Phase 1 --- Frontend scaffold
**Goal:** Add the smallest working React + Vite application.

Tasks:
- Create `frontend/` with the standard Vite React setup.
- Add only the necessary files, such as `src/main.jsx`, `src/App.jsx`,
`src/App.css`, and `src/api.js`.
- Add a simple page shell with the Recall name and clear sections for backend status, vault ingestion, and
chat.
- Add a configurable API base URL with a local default based on the verified project configuration.
- Keep styling minimal and readable.

Restrictions:
- Do not connect the chat or ingestion actions yet.
- Do not change backend files or dependencies.

**Exit criteria:** The frontend starts locally and renders without
errors.

## Phase 2 --- Backend health status
**Goal:** Make backend connectivity visible.

Tasks:
- Add an API helper for the existing health endpoint.
- Show a small status: checking, connected, or unavailable.
- Handle network errors gracefully.
- Use the actual endpoint and response format verified in Phase 0.

**Exit criteria:** The UI accurately reports whether the backend is
reachable.

## Phase 3 --- Vault ingestion controls
**Goal:** Let the user trigger ingestion of the already-configured
vault.

Tasks:
- Add an **Ingest vault** button.
- Call the existing ingestion endpoint using its exact request format.
- Show loading, success, and error states.
- Display returned ingestion statistics when available.
- Show the configured vault path only if it is safely and actually
available to the frontend; otherwise display a clear note that the path
is configured on the backend before startup.

Restrictions:
- Do not create a fake folder picker.
- Do not promise runtime vault switching if the backend does not support it.
- Do not change backend behavior to make ingestion easier.

**Exit criteria:** The user can trigger ingestion and understand the
outcome.

## Phase 4 --- Chat and source citations
**Goal:** Ask questions and read answers in the UI.

Tasks:
- Add a text input or textarea and a **Send** button.
- Call the existing chat endpoint using the exact request schema.
- Display the submitted question and generated answer.
- Display source filenames and heading/citation metadata using the actual response fields.
- Handle empty questions, loading, failed requests, and missing sources.
- Keep the interface simple; no conversation persistence or streaming.

**Exit criteria:** The user can ask a question and see the answer and
any citations returned by the backend.

## Phase 5 --- End-to-end verification and cleanup
**Goal:** Verify the full user journey and finish the frontend.

Test this flow:
1. Start the existing backend and Qdrant as documented.
2. Start the frontend.
3. Confirm backend status.
4. Trigger ingestion of the configured vault.
5. Ask a question that should be answerable from the fixture or vault notes.
6. Confirm the answer and source/heading metadata appear.
7. Test an empty question, a failed request, and an unavailable backend if practical.
8. Confirm the frontend does not contain secrets.
9. Inspect the final diff and verify no backend files changed.

Add only essential run instructions to the frontend README or root README if needed.

**Exit criteria:** The UI works locally for the core workflow, errors
are understandable, and the final report identifies any remaining
limitations.

---

## Suggested final structure
Adapt this to the existing repository rather than reorganizing working
backend files:

```
recall-rag/
├── [existing backend files — unchanged]
├── [existing tests — unchanged]
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── api.js
    │   └── main.jsx
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Definition of done

- The React frontend runs locally.
- Backend health status is visible.
- The user can trigger ingestion for the already-configured vault.
- The user can ask a question through the existing chat endpoint.
- The answer and returned source/heading citations are displayed.
- Loading and error states are understandable.
- Backend code and tests remain untouched.
- The agent stopped at the end of each phase and waited for approval.
**Recommended order:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 →
Phase 5.

**Important:** Complete and review each phase before proceeding. Do not
ask the coding agent to implement all phases at once.
