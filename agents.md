AGENTS.md — Recall
Project purpose
Recall (recall-rag) is a learning-focused naive RAG assistant that indexes an Obsidian Markdown vault, retrieves relevant notes using Qdrant, and answers questions with source-file and heading-level citations.

The developer is learning AI engineering while building this project. Optimize for understanding, correctness, and maintainability, not maximum code generation speed. Recall is a learning and portfolio project, not a claim of production readiness.
How to work with the developer
Act as a programming mentor and pair programmer—not just a code generator.

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
  Recall already has an end-to-end naive RAG pipeline: Markdown ingestion and heading-aware chunking, embeddings and Qdrant storage, retrieval, Gemini answer generation, source/heading metadata, a FastAPI backend, and a React frontend. Local embeddings using sentence-transformers/all-MiniLM-L6-v2 are being used to avoid relying on Gemini embedding quota; Gemini remains the answer-generation provider.

Do not assume every item above is currently correct or fully reliable. Inspect the current implementation and tests before changing it. The current priority is retrieval quality and grounded answers, not adding features or optimizing latency prematurely.
Agreed retrieval-quality roadmap
Work through these milestones in order, one at a time. Do not skip ahead or implement the entire roadmap in one pass.

1. Read-only diagnosis and baseline
   - Inspect current Git status/diff and relevant retrieval, chunking, prompting, citation, and test code.
   - Review the existing scoped-retrieval implementation and preserve it; do not rewrite it from scratch.
   - Reproduce known issues using synthetic/in-memory fixtures only.
   - Confirm the known mismatch where a user mentions WEEKLY GOALS.md but the indexed filename is decorated, e.g. 🗓️ WEEKLY GOALS.md.
   - Inspect how scoped results are merged with semantic results and how citations are constructed.
   - Identify what is known versus still uncertain. Do not edit code in this milestone.
   - Report the smallest proposed fix and test plan, then wait for approval before implementation.
2. Fix known file/heading resolution bugs
   - Resolve a requested filename against indexed source paths robustly, including decorative emoji/prefix differences where the match is unique.
   - Preserve ambiguity safety: never silently select one of multiple plausible files.
   - Preserve correct path handling and avoid broad fuzzy matching that could resolve to the wrong note.
   - Test exact names, decorated names, missing names, and ambiguous candidates.
   - Keep heading-reference parsing and nested section retrieval working, including Unicode and curly/straight quotes.
   - Make only the smallest approved changes; inspect the diff and run focused tests before reporting.
3. Create a repeatable retrieval evaluation set
   - Use synthetic Markdown notes, not the real vault, to create approximately 15–30 representative questions with expected source files/chunks.
   - Cover explicit filename and heading references, natural/implicit references, paraphrased wording, multi-chunk or multi-note questions, ambiguous/missing references, and unanswerable questions.
   - Measure whether expected evidence appears in the retrieved results (for example Recall@k), ranking quality where practical, and irrelevant-context inclusion.
   - Establish and record a baseline before changing retrieval strategy.
   - Keep evaluation deterministic and affordable; do not require Gemini calls for basic retrieval metrics.
4. Improve the existing retrieval pipeline
   - Use evaluation results to investigate ranking, chunk boundaries, parent/child section expansion, retrieval limits, metadata filters, deduplication, and context selection.
   - Preserve the local embedding model and current Qdrant collection initially; do not mix model changes, re-indexing, chunking changes, and retrieval-strategy changes in one experiment.
   - Do not assume every query should use scoped retrieval. Natural-language questions without explicit references must still work through general retrieval.
   - Keep changes small and compare against the baseline.
5. Evaluate hybrid search as a controlled experiment
   - Only after the baseline and known bugs are addressed, compare current dense retrieval with dense + sparse retrieval using Qdrant and Reciprocal Rank Fusion (RRF), if evaluation indicates a need.
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
- Do not promise “perfect retrieval.” Define a test set, establish a baseline, set realistic targets, and report measured results.
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
