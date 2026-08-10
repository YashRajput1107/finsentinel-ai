from pathlib import Path

from src.rag.chunking import parse_metadata


def test_parse_metadata_sec_filing():
    meta = parse_metadata(Path("AAPL_10K_2024-11-01.txt"))
    assert meta["ticker"] == "AAPL"
    assert meta["source_type"] == "filing"
    assert meta["form_type"] == "10K"
    assert meta["doc_date"] == "2024-11-01"


def test_parse_metadata_transcript():
    meta = parse_metadata(Path("AAPL_2016-01-26.txt"))
    assert meta["ticker"] == "AAPL"
    assert meta["source_type"] == "transcript"
    assert meta["form_type"] == "earnings_call"