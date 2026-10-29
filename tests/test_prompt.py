"""Tests for the RAG prompt builder."""

import pytest

from app.models import Chunk, SearchResult
from app.prompt import build_rag_prompt


def _make_result(
    source: str, headings: list[str], content: str, score: float = 0.9, chunk_index: int = 0
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            source_path=source,
            heading_path=headings,
            chunk_index=chunk_index,
            content=content,
        ),
        score=score,
    )



def test_prompt_contains_system_instruction() -> None:
    results = [_make_result("notes/a.md", ["Section"], "Some content.")]
    prompt = build_rag_prompt("What is this?", results)
    assert "personal knowledge assistant" in prompt
    assert "ONLY the context" in prompt or "primary source of truth" in prompt
    assert "I don't have enough information to answer that." in prompt


def test_prompt_contains_question() -> None:
    results = [_make_result("notes/a.md", ["Section"], "Some content.")]
    prompt = build_rag_prompt("What projects am I working on?", results)
    assert "What projects am I working on?" in prompt


def test_prompt_contains_numbered_references() -> None:
    results = [
        _make_result("projects/recall.md", ["Recall", "MVP"], "Recall is a RAG assistant."),
        _make_result("career/jobs.md", ["Targets"], "Target companies list."),
    ]
    prompt = build_rag_prompt("What am I building?", results)
    assert "[1] projects/recall.md | Recall > MVP" in prompt
    assert "[2] career/jobs.md | Targets" in prompt
    assert "Recall is a RAG assistant." in prompt
    assert "Target companies list." in prompt


def test_prompt_uses_root_label_for_no_headings() -> None:
    results = [_make_result("notes/orphan.md", [], "Orphan content with no heading.")]
    prompt = build_rag_prompt("Tell me about orphans.", results)
    assert "(root)" in prompt


def test_prompt_structure_order() -> None:
    """CONTEXT must appear before QUESTION, which must appear before ANSWER."""
    results = [_make_result("notes/a.md", ["A"], "Content A.")]
    prompt = build_rag_prompt("My question?", results)
    context_pos = prompt.index("--- CONTEXT ---")
    question_pos = prompt.index("--- QUESTION ---")
    answer_pos = prompt.index("--- ANSWER ---")
    assert context_pos < question_pos < answer_pos


def test_prompt_empty_question_raises() -> None:
    results = [_make_result("notes/a.md", ["A"], "Content.")]
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_rag_prompt("   ", results)


def test_prompt_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="No search results provided"):
        build_rag_prompt("What is this?", [])


def test_prompt_strips_whitespace_from_question() -> None:
    results = [_make_result("notes/a.md", ["A"], "Content.")]
    prompt = build_rag_prompt("  What is this?  ", results)
    assert "What is this?" in prompt


def test_prompt_sorts_chunks_sequentially_per_document() -> None:
    # doc_a has chunk 2 (score 0.95) and chunk 0 (score 0.8)
    # doc_b has chunk 1 (score 0.7)
    # Expected order:
    # [1] doc_a chunk 0
    # [2] doc_a chunk 2
    # [3] doc_b chunk 1
    results = [
        _make_result("notes/doc_a.md", ["Part 2"], "Doc A Part 2", score=0.95, chunk_index=2),
        _make_result("notes/doc_b.md", ["Part 1"], "Doc B Part 1", score=0.70, chunk_index=1),
        _make_result("notes/doc_a.md", ["Part 0"], "Doc A Part 0", score=0.80, chunk_index=0),
    ]
    prompt = build_rag_prompt("Explain everything", results)

    pos_a0 = prompt.index("Doc A Part 0")
    pos_a2 = prompt.index("Doc A Part 2")
    pos_b1 = prompt.index("Doc B Part 1")

    # Document A comes before Document B (higher best score), and A's chunks are in reading order
    assert pos_a0 < pos_a2 < pos_b1
    assert "[1] notes/doc_a.md" in prompt
    assert "[2] notes/doc_a.md" in prompt
    assert "[3] notes/doc_b.md" in prompt
