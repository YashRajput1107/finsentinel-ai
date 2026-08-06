import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.rag.embed_index import search, EMBED_MODEL

GOLD_PATH = Path(__file__).parent / "gold_questions.json"


def evaluate(ks=(1, 3, 5)) -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBED_MODEL)
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    rows = []

    for item in gold:
        # IMPORTANT: no ticker filter here — we TEST whether retrieval finds the
        # right company on its own. Filtering by ticker would make hits@k a trivial 1.0.
        res = search(item["question"], k=max_k, model=model)
        got = res["ticker"].tolist()
        # rank (1-based) of the first chunk from the expected company, or None
        rank = next((i + 1 for i, t in enumerate(got) if t == item["expected_ticker"]), None)
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
        rows.append({"expected": item["expected_ticker"], "rank": rank, "top_tickers": got})

    n = len(gold)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    for k in ks:
        print(f"hits@{k}: {hits[k]}/{n} = {hits[k] / n:.2f}")


if __name__ == "__main__":
    evaluate()