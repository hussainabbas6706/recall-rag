"""Prompt construction for grounded answer generation."""

from collections.abc import Sequence

from app.models import ChatMessage, SearchResult

INSUFFICIENT_CONTEXT_ANSWER = "I don't have enough information to answer that."

_SYSTEM_INSTRUCTION = (
    "You are a personal knowledge assistant grounded in the user's notes. "
    "The context below contains retrieved excerpts from the user's vault. "
    "Use this context as your primary source of truth for vault-specific facts "
    "(dates, tasks, goals, names, schedules, and other concrete details). "
    "When the user asks a question that relates to topics covered in the context, "
    "you may reason, give opinions, and draw on general knowledge to provide a helpful answer — "
    "but always anchor your response in the context and cite the sources you used "
    "by their reference numbers, e.g. [1] or [1][2]. "
    "Use the conversation history to resolve references and follow-up questions; "
    "do not treat prior assistant messages as factual sources. "
    "If the question is completely unrelated to anything in the provided context, say exactly: "
    f"\"{INSUFFICIENT_CONTEXT_ANSWER}\" "
    "Do not invent vault-specific facts that are not in the context."
)


def sort_results_sequentially(results: list[SearchResult]) -> list[SearchResult]:
    """Re-order results for sequential, coherent LLM context.

    Files are ranked by their highest similarity score (most relevant first).
    Within each file, chunks are sorted by chunk_index (document reading order).
    This ensures the LLM reads each source as a continuous narrative rather
    than disconnected fragments, reducing the chance of hallucination.
    """
    # Track the best score seen for each source file
    best_score: dict[str, float] = {}
    for result in results:
        path = result.chunk.source_path
        if path not in best_score or result.score > best_score[path]:
            best_score[path] = result.score

    return sorted(
        results,
        key=lambda r: (
            -best_score[r.chunk.source_path],  # file relevance: highest score first
            r.chunk.chunk_index,               # reading order within the file
        ),
    )


def _format_context_block(results: list[SearchResult]) -> str:
    """Format search results into a numbered, sequentially-ordered reference list."""
    ordered = sort_results_sequentially(results)
    lines: list[str] = []
    for i, result in enumerate(ordered, start=1):
        chunk = result.chunk
        breadcrumb = " > ".join(chunk.heading_path) if chunk.heading_path else "(root)"
        lines.append(f"[{i}] {chunk.source_path} | {breadcrumb}")
        lines.append(chunk.content.strip())
        lines.append("")  # blank line between entries
    return "\n".join(lines).strip()


def build_rag_prompt(
    question: str,
    results: list[SearchResult],
    history: Sequence[ChatMessage] = (),
    context_note: str | None = None,
) -> str:
    """Build a grounded RAG prompt from a question and retrieved chunks.

    Args:
        question: The user's natural language question.
        results: Ranked search results from Qdrant, ordered best-first.
        history: Previous chat messages used to interpret follow-up questions.
        context_note: Optional retrieval limitation to disclose to the model.

    Returns:
        A formatted prompt string ready to send to the LLM.

    Raises:
        ValueError: If question is empty or results list is empty.
    """
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")
    if not results:
        raise ValueError(
            "No search results provided. Run a search before building a prompt."
        )

    context_block = _format_context_block(results)
    history_block = "\n".join(
        f"{message.role.title()}: {message.content}" for message in history
    )
    history_section = (
        f"--- CONVERSATION HISTORY ---\n\n{history_block}\n\n"
        if history_block
        else ""
    )
    context_note_section = (
        f"--- RETRIEVAL NOTE ---\n\n{context_note.strip()}\n\n"
        if context_note and context_note.strip()
        else ""
    )

    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"--- CONTEXT ---\n\n"
        f"{context_block}\n\n"
        f"{context_note_section}"
        f"{history_section}"
        f"--- QUESTION ---\n\n"
        f"{clean_question}\n\n"
        f"--- ANSWER ---"
    )
