import logging

import ollama
from sentence_transformers import SentenceTransformer

from src.rag.embed_index import search, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_MODEL = "llama3.2"

# This system prompt IS the grounding mechanism — the one soft guardrail we discussed.
SYSTEM_PROMPT = (
    "You are a financial document assistant. Answer the user's question using ONLY "
    "the numbered context passages provided. Cite the passages you use by their number "
    "in square brackets, e.g. [1] or [2]. "
    "If the answer is not contained in the context, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "Do not use any outside knowledge."
)
def format_context(hits) -> str:
    """Number each retrieved chunk and label its source — this enables citations."""
    lines=[]
    for i, (_, r) in enumerate(hits.iterrows(), start=1):
        source = f"{r.ticker} {r.form_type} {r.doc_date}"
        lines.append(f"[{i}] ({source})\n{r.text}")
    return "\n\n".join(lines)

def answer(question: str, ticker=None, k: int = 5,
           model: SentenceTransformer | None = None, use_mmr: bool = True) -> dict:
    hits = search(question, k=k, ticker=ticker, model=model, use_mmr=use_mmr)
    if len(hits) == 0:                       # nothing retrieved -> can't ground -> refuse
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    context = format_context(hits)
    user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
        options={"temperature": 0.0},        # deterministic -> less creative -> less hallucination
    )

    sources = [
        {"n": i, "ticker": r.ticker, "form_type": r.form_type,
         "doc_date": r.doc_date, "chunk_id": int(r.chunk_id)}
        for i, (_, r) in enumerate(hits.iterrows(), start=1)
    ]
    return {"answer": resp["message"]["content"], "sources": sources}

if __name__ == "__main__":
    m = SentenceTransformer(EMBED_MODEL)

    print("=" * 80, "\nGROUNDED QUESTION (answer IS in the docs)\n")
    out = answer("What are the main risks Tesla faces in vehicle production?",
                 ticker="TSLA", model=m)
    print(out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['ticker']} {s['form_type']} {s['doc_date']} #{s['chunk_id']}")

    print("\n", "=" * 80, "\nHONESTY TEST (answer is NOT in the docs)\n")
    out2 = answer("What is Tesla's exact quarterly cash dividend per share?",
                  ticker="TSLA", model=m)
    print(out2["answer"])