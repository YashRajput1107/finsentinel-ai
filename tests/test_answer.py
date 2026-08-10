from unittest.mock import patch

import pandas as pd

from src.rag.answer import answer
from src.rag.evaluate_answers import is_refusal, has_citation


def test_is_refusal():
    assert is_refusal("I don't know based on the provided documents.") is True
    assert is_refusal("Tesla faces risks [1].") is False


def test_has_citation():
    assert has_citation("grounded [2] answer") is True
    assert has_citation("no citation") is False


@patch("src.rag.answer.ollama.chat")
@patch("src.rag.answer.search")
def test_answer_wires_and_returns_sources(mock_search, mock_chat):
    # Arrange — fake the retrieval and the LLM
    mock_search.return_value = pd.DataFrame({
        "ticker": ["TSLA"], "source_type": ["filing"], "form_type": ["10K"],
        "doc_date": ["2025-01-01"], "chunk_id": [42], "text": ["Tesla risk text."],
        "score": [0.7],
    })
    mock_chat.return_value = {"message": {"content": "Tesla faces risks [1]."}}

    # Act
    out = answer("What are Tesla's risks?", ticker="TSLA")

    # Assert — OUR wiring, not the model's quality
    assert out["answer"] == "Tesla faces risks [1]."
    assert out["sources"][0]["chunk_id"] == 42
    assert mock_chat.called


@patch("src.rag.answer.search")
def test_answer_refuses_when_nothing_retrieved(mock_search):
    mock_search.return_value = pd.DataFrame(columns=["ticker", "text"])  # empty
    out = answer("obscure", ticker="TSLA")
    assert "don't know" in out["answer"].lower()
    assert out["sources"] == []