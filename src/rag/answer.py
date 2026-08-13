import logging
import time

import ollama
from sentence_transformers import SentenceTransformer

from src.rag.embed_index import search, EMBED_MODEL
from src.utils.config import LLM_PROVIDER, GROQ_API_KEY, OLLAMA_MODEL, GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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

def chat_completion(system: str, user: str, provider: str | None = None) -> str:
    """One interface, two backends - ollama locally, groq when deployed.
    The rest of the code doesn't know or care which one is behind this."""
    provider = provider or LLM_PROVIDER
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    started = time.perf_counter()

    try:
        if provider == "groq":
            from groq import Groq
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set")
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=0.0)
            text, model = resp.choices[0].message.content, GROQ_MODEL
        else:
            resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                               options={"temperature": 0.0})  # deterministic -> less hallucination
            text, model = resp["message"]["content"], OLLAMA_MODEL

        logger.info("llm ok | provider=%s model=%s %.1fs",
                    provider, model, time.perf_counter() - started)
        return text

    except Exception as e:
        # log the real cause for us, hand the user something safe to read
        logger.error("llm failed | provider=%s after %.1fs: %s",
                     provider, time.perf_counter() - started, e)
        return "The assistant is temporarily unavailable. Please try again."


def answer(question: str, ticker=None, k: int = 5,
           model: SentenceTransformer | None = None, use_mmr: bool = True) -> dict:
    hits = search(question, k=k, ticker=ticker, model=model, use_mmr=use_mmr)
    if len(hits) == 0:                       # nothing retrieved -> can't ground -> refuse
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    context = format_context(hits)
    user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    text = chat_completion(SYSTEM_PROMPT, user_msg)

    sources = [
        {"n": i, "ticker": r.ticker, "form_type": r.form_type,
         "doc_date": r.doc_date, "chunk_id": int(r.chunk_id)}
        for i, (_, r) in enumerate(hits.iterrows(), start=1)
    ]
    return {"answer": text, "sources": sources}

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