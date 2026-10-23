"""Unit tests for answer generation module."""

from unittest.mock import MagicMock

import pytest

from app.generator import generate_answer, get_chat_model


def test_get_chat_model_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_chat_model(api_key="")


def test_generate_answer_empty_prompt_raises() -> None:
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generate_answer("   ")


def test_generate_answer_calls_model_with_prompt() -> None:
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Recall is a local RAG assistant. [1]"
    mock_model.invoke.return_value = mock_response

    prompt = "--- CONTEXT ---\n[1] file.md | Title\nContent\n--- QUESTION ---\nWhat is Recall?\n--- ANSWER ---"
    answer = generate_answer(prompt, model=mock_model)

    assert answer == "Recall is a local RAG assistant. [1]"
    mock_model.invoke.assert_called_once_with(prompt)


def test_generate_answer_handles_list_content() -> None:
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = ["Recall ", {"text": "is grounded."}, " [1]"]
    mock_model.invoke.return_value = mock_response

    answer = generate_answer("A valid prompt", model=mock_model)
    assert answer == "Recall is grounded. [1]"


def test_generate_answer_strips_output_whitespace() -> None:
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "   Cleaned answer text.   \n"
    mock_model.invoke.return_value = mock_response

    answer = generate_answer("A valid prompt", model=mock_model)
    assert answer == "Cleaned answer text."
