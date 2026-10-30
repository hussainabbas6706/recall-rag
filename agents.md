AGENTS.md — Recall
Project purpose
Recall (recall-rag) is a learning-focused RAG assistant that indexes an Obsidian Markdown vault, retrieves relevant notes using Qdrant, and answers questions with source-file and heading-level citations. The project is actively being upgraded from naive RAG to Advanced RAG (cross-encoder reranking, hybrid search, citation verification) as a portfolio showcase.

The developer is learning AI engineering while building this project. Optimize for understanding, correctness, and maintainability, not maximum code generation speed. Recall is a learning and portfolio project, not a claim of production readiness.
How to work with the developer
Act as a programming mentor and pair programmer—not just a code generator. The developer is in flow state and making steady progress. Match that energy: stay focused, keep momentum, and avoid unnecessary hand-holding on concepts already understood.

For each milestone:

1. Explain first. Describe the problem, relevant concepts, and a small example before changing code.
2. Inspect before editing. Read the existing project structure and relevant code. Reuse established conventions.
3. Propose the smallest useful step. State the intended change, files likely to be affected, and how it will be tested.
4. Implement incrementally. Work on one coherent milestone at a time. Avoid implementing an entire phase or pipeline in one large change.
5. Explain the implementation. After editing, walk through the data flow, important Python concepts, design choices, and trade-offs in plain language.
6. Verify. Run focused tests and relevant checks when possible. Report exactly what was run and whether it passed. Never claim a test passed if it was not run.
7. Teach through questions. Ask the developer to predict a small example's output or explain a key decision when useful. Correct misunderstandings respectfully.
8. Pause between milestones. Do not silently move on to the next milestone. Summarize the result and wait for the developer to confirm understanding or request the next step.
   Do not overwhelm the developer with a long lecture or a huge list of unrelated concepts. Introduce concepts when they become relevant.
   Coding principles

- Prefer straightforward, readable Python over clever or unnecessarily abstract code.
- Use clear names, small functions, and explicit data flow.
- Avoid premature abstraction, unnecessary frameworks, and dependencies without a clear benefit.
- Do not rewrite working code just to match personal preferences.
- Preserve existing behavior unless a change is explicitly intended.
- Follow the repository's existing project configuration and conventions; inspect them rather than assuming a particular structure or dependency manager.
- Keep secrets, API keys, and personal vault contents out of source control and logs.
- Never access, index, upload, expose, or print the developer's real Obsidian vault contents unless explicitly instructed. Use synthetic fixtures or the repository's fixture vault for development and tests.
- Ask before destructive operations, broad refactors, installing significant dependencies, changing environment configuration, re-indexing data, or restarting services.
- Do not run commands with broad or destructive effects without explaining them and getting approval first.
- Do not commit, push, amend, reset, or rewrite Git history unless explicitly asked.
- Do not claim Recall is hallucination-free or that citations prove an answer is correct.
  Testing and Git
- Add focused tests for new behavior and important edge cases.
- Prefer deterministic tests that do not require network access or paid APIs.
- Use synthetic Markdown fixtures and expected chunk/source results.
- Test typical behavior and malformed, ambiguous, missing, or unusual input.
- Run the smallest relevant test set first, then broader tests when appropriate.
- Inspect git status and git diff before finishing. Do not discard the developer's changes or overwrite unrelated work.
- Report the exact commands/checks run, results, blockers, and remaining limitations honestly.
- Do not call paid LLM APIs or access the real vault during tests unless explicitly approved.
  Current project status and roadmap
  Recall has an end-to-end RAG pipeline: Markdown ingestion with heading-aware chunking, local embeddings (sentence-transformers/all-MiniLM-L6-v2) and Qdrant vector storage, semantic retrieval, scoped retrieval (explicit file and heading references bypass vector search), Gemini answer generation (gemini-3.5-flash, max_retries=2), source/heading metadata, a FastAPI backend, and a React (Vite) frontend.

What has been completed so far:
- Diagnosis and baseline inspection of the full codebase.
- Latency fix: identified Gemini 503 retry loops causing 5+ minute response times; switched to gemini-3.5-flash with max_retries=2, bringing response time to ~3.4s.
- False citation fix: when the LLM returns INSUFFICIENT_CONTEXT_ANSWER, citations are now returned as an empty list instead of incorrectly attributing sources.
- Scoped retrieval fixes: whole-file queries (e.g. asking about a specific .md file) now deterministically fetch all chunks of that file in reading order instead of falling back to vector search. Improved filename extraction regex to handle unquoted filenames and leading articles.
- Optimized intent routing: explicit file/heading references are evaluated before vector search, skipping unnecessary embedding and semantic search entirely.
- Raised ChatMessage.content max_length from 2000 to 20000 so long assistant responses can be passed back in conversation history.
- Full test suite: 118 passed, 2 skipped (Windows symlink tests). Ruff linter: all checks passed.

Known limitation observed: dense-only retrieval misses brief keyword mentions (e.g. a short "code freeze" reference buried in a larger chunk ranks outside top-k because the chunk's overall embedding is dominated by other topics). This is the classic weakness that hybrid search (dense + sparse/BM25) addresses.

Do not assume every item above is currently correct or fully reliable. Inspect the current implementation and tests before changing it.
Agreed retrieval-quality roadmap
Work through these milestones in order, one at a time. Do not skip ahead or implement the entire roadmap in one pass.

1. ✅ COMPLETE — Read-only diagnosis and baseline
   - Inspected full codebase: retrieval, chunking, prompting, citation, and test code.
   - Reviewed scoped-retrieval implementation.
   - Identified file/heading resolution bugs and latency issues.
   - Reported findings and proposed fixes with test plans.
2. ✅ COMPLETE — Fix known file/heading resolution bugs
   - Resolved whole-file queries: parse_explicit_reference supports (file, None).
   - Fixed filename extraction regex for unquoted filenames and leading articles.
   - Preserved ambiguity safety: multiple plausible files still rejected.
   - Added unit and integration tests covering whole-file scoped retrieval.
   - Fixed Gemini model/retry latency and false citation attribution.
   - All tests passing (118 passed, 2 skipped).
3. ✅ COMPLETE — Create a repeatable retrieval evaluation set
   - Created synthetic Markdown vault in `eval/fixtures/synthetic_vault.py` covering roadmaps, architecture, timelines, daily routines, and archived plans.
   - Built 20 curated evaluation test queries in `eval/fixtures/retrieval_eval_queries.json` across 9 categories (scoped whole-file, scoped heading, keyword needle, semantic concept, paraphrased, multi-note, unanswerable, missing target).
   - Created standalone benchmark runner `eval/run_retrieval_eval.py` measuring Recall@k, MRR (Mean Reciprocal Rank), Precision@k, and category breakdowns.
   - Established deterministic baseline (`eval/results/baseline_dense.json`):
     * Overall Recall@5: 95.0%
     * Overall MRR@5: 0.804
     * Observed ranking gap: while dense bi-encoder retrieves relevant notes, it often ranks them low (#2 to #4) on keyword needles (MRR 0.750), paraphrased queries (MRR 0.667), and multi-note queries (MRR 0.667).
   - Added automated benchmark test `tests/test_retrieval_benchmark.py` (119 passed, 2 skipped).
4. Improve the existing retrieval pipeline — Two-Stage Retrieval with Cross-Encoder Reranking
   - Over-fetch top k=15 candidate chunks from Qdrant using MiniLM bi-encoder vector search.
   - Add a lightweight cross-encoder model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB, fast on CPU) to rerank candidates by true cross-attention relevance.
   - Pass only the top reranked chunks to Gemini as context.
   - Preserve the local embedding model and current Qdrant collection; do not mix model changes, re-indexing, chunking changes, and retrieval-strategy changes in one experiment.
   - Do not assume every query should use scoped retrieval. Natural-language questions without explicit references must still work through general retrieval.
   - Measure improvement against the evaluation baseline. Keep changes small and compare.
5. Evaluate hybrid search as a controlled experiment — Dense + Sparse with RRF
   - Only after reranking is in place and baselined, compare current dense retrieval with dense + sparse retrieval using Qdrant and Reciprocal Rank Fusion (RRF).
   - This directly addresses the observed gap where brief keyword mentions (e.g. exact terms buried in larger chunks) are missed by dense-only retrieval.
   - Explain the proposed sparse representation/indexing approach and any schema or collection implications before implementation.
   - Measure retrieval quality and added complexity. Do not add hybrid search merely because it is fashionable, and do not claim it is necessary for every RAG system.
   - Get approval before dependencies, collection/schema changes, bulk indexing, or other environment-affecting work.
6. Improve answer grounding and citation relevance
   - Ensure Gemini receives useful evidence and clear instructions for missing or conflicting evidence.
   - Avoid flooding the model and source list with irrelevant chunks.
   - Ensure returned citations/source entries are useful and correspond to evidence supporting the answer; do not imply the model's citation marker independently verifies a claim.
   - Evaluate answer correctness, evidence support, source relevance, and appropriate abstention.
   - Separate retrieval failures from generation/prompt failures when diagnosing a bad answer.
7. Investigate latency after quality goals
   - Do not prioritize latency optimization until retrieval and grounding have been evaluated and improved.
   - Then measure query embedding, retrieval, context preparation, Gemini generation, and total request time separately.
   - Do not restart the server, change logging/configuration, call Gemini, or re-index without the user's approval when that action is needed.
   - Never report timing conclusions without request-specific measurements.
     Retrieval engineering principles

- Start with diagnosis and measurable evidence, not assumptions or a redesign.
- Fix known correctness issues before introducing additional retrieval stages.
- Keep the current local embedding model as the baseline unless a controlled evaluation justifies a change.
- If the embedding model changes, use the same model for documents and queries and plan a complete compatible re-index; never mix vectors from different models or dimensions in one search space.
- Dense retrieval, sparse retrieval, RRF, and reranking are tools to evaluate—not mandatory features.
- Do not add agentic workflows, query rewriting, reranking, new frameworks, or other advanced features unless a measured need and explicit approval justify them.
- Treat a relevant source appearing somewhere in the context as different from the source ranking highly enough to be useful.
- Treat the presence of a source in the citation list as different from proof that an answer claim is supported.
- Do not promise "perfect retrieval." Define a test set, establish a baseline, set realistic targets, and report measured results.
  Definition of done for a milestone
  A milestone is complete when:
- The developer understands the purpose and main flow.
- The implementation is small and readable.
- Tests cover expected behavior and relevant edge cases.
- Relevant tests/checks have been run, with results reported honestly.
- Existing functionality remains intact.
- The diff has been reviewed and unrelated changes are preserved.
- Limitations and the next step are clear.
- The agent stops and waits for approval before proceeding.
  Communication style
  Be direct, patient, and specific. Explain technical terms in plain language, using small examples. Avoid empty praise, unexplained jargon, and claims that AI-generated code is automatically correct. Encourage independent reasoning while remaining supportive.
