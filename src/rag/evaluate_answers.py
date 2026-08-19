import json
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.rag.embed_index import EMBED_MODEL
from src.rag.answer import answer

GOLD_PATH = Path(__file__).parent / "gold_questions.json"
REVIEW_PATH = Path(__file__).parent / "answer_review.csv"
REFUSAL = "i don't know based on the provided documents"
# what the app shows when the LLM call itself fails. that's an infrastructure problem,
# not the model behaving badly, so it must not be scored as a wrong answer.
API_ERROR = "temporarily unavailable"


def is_refusal(text: str) -> bool:
    return REFUSAL in text.strip().lower()


def is_api_error(text: str) -> bool:
    return API_ERROR in text.strip().lower()


def has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


def evaluate() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBED_MODEL)

    refuse_total = refuse_ok = 0        # unanswerable set
    ans_total = cite_ok = wrong_refuse = 0   # answerable set
    api_errors = 0
    review = []

    for item in gold:
        q = item["question"]
        answerable = item.get("answerable", True)
        tkr = item.get("ticker", item.get("expected_ticker"))
        out = answer(q, ticker=tkr, model=model)      # full pipeline (MMR on)
        text = out["answer"]

        if is_api_error(text):
            # the call never reached the model - scoring this would measure Groq's
            # uptime, not the pipeline's behaviour
            api_errors += 1
            print(f"[API ERROR - excluded] {q}\n")
            continue

        refused = is_refusal(text)

        if not answerable:
            refuse_total += 1
            refuse_ok += int(refused)
            print(f"[SHOULD REFUSE] {q}\n  refused? {refused}\n  -> {text[:140]}\n")
        else:
            ans_total += 1
            wrong_refuse += int(refused)          # answerable Q it wrongly punted on
            cite_ok += int(has_citation(text))
            srcs = ", ".join(f"{s['ticker']} {s['form_type']} {s['doc_date']}" for s in out["sources"])
            print(f"[ANSWERABLE] {q}\n  cites? {has_citation(text)}  wrongly_refused? {refused}")
            print(f"  ANSWER: {text[:300]}\n  SOURCES: {srcs}\n")
            review.append({"question": q, "answer": text, "sources": srcs,
                           "grounded(y/n)": "", "relevant(y/n)": ""})

    # human-judgment sheet for the answerable set
    pd.DataFrame(review).to_csv(REVIEW_PATH, index=False, encoding="utf-8")

    print("=" * 60)
    print(f"Refusal accuracy (unanswerable): {refuse_ok}/{refuse_total}")
    print(f"Citation rate    (answerable)  : {cite_ok}/{ans_total}")
    print(f"Wrong refusals   (answerable)  : {wrong_refuse}/{ans_total}")
    if api_errors:
        print(f"API errors (excluded from scores): {api_errors}")
    print(f"\nHuman-judge groundedness/relevance in: {REVIEW_PATH.name}")


if __name__ == "__main__":
    evaluate()