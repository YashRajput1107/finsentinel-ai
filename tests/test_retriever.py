import numpy as np
import pytest

from src.rag.embed_index import _build_selector, _mmr, search


def test_selector_none_without_filter(sample_meta):
    sel, ids = _build_selector(sample_meta)
    assert sel is None and ids is None


def test_selector_filters_ticker(sample_meta):
    _, ids = _build_selector(sample_meta, ticker="TSLA")
    assert sorted(ids.tolist()) == [0, 1]


def test_selector_unknown_ticker_is_empty(sample_meta):
    _, ids = _build_selector(sample_meta, ticker="ZZZZ")   # not in universe
    assert len(ids) == 0                                    # empty, not a crash


def test_mmr_seeds_most_relevant_then_avoids_duplicate():
    rel = np.array([0.90, 0.88, 0.70])          # 1 is near-dup of 0; 2 is distinct
    sim = np.array([[1.00, 0.99, 0.10],
                    [0.99, 1.00, 0.10],
                    [0.10, 0.10, 1.00]])
    order = _mmr(rel, sim, k=2, lambda_=0.6)
    assert order[0] == 0        # most relevant picked first
    assert 2 in order and 1 not in order   # distinct chosen over the duplicate


def test_mmr_never_selects_duplicate():
    # regression: with lambda=1 the old 'is' bug would pick index 0 twice
    rel = np.array([0.9, 0.1])
    sim = np.array([[1.0, 0.0], [0.0, 1.0]])
    order = _mmr(rel, sim, k=2, lambda_=1.0)
    assert len(set(order)) == len(order)


def test_search_rejects_bad_input():
    with pytest.raises(ValueError):
        search("   ")           # empty query
    with pytest.raises(ValueError):
        search("hello", k=0)    # non-positive k