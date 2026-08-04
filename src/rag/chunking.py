import logging
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import PROCESSED_SEC_DIR, TRANSCRIPTS_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "rag_chunks.parquet"

# chunk_size is in CHARACTERS. ~800 chars ≈ 180–200 tokens, which stays safely
# under all-MiniLM-L6-v2's 256-token input limit (Day 19). If a chunk exceeded
# that limit, the embedder would SILENTLY TRUNCATE it and we'd lose text.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def parse_metadata(path:Path)->dict:
    parts=path.stem.split("_")
    ticker=parts[0]
    if len(parts)==3:
        return {"ticker": ticker, "source_type": "filing",
                "form_type": parts[1], "doc_date": parts[2], "doc_id": path.name}
    return {"ticker": ticker, "source_type": "transcript",     # earnings call
            "form_type": "earnings_call", "doc_date": parts[1], "doc_id": path.name}


def iter_documents():
    """Yield (text, metadata) for every SEC filing and transcript."""
    for folder in (PROCESSED_SEC_DIR, TRANSCRIPTS_DIR):
        for path in sorted(folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:                         # skip any empty file defensively
                yield text, parse_metadata(path)

def build_chunks() -> pd.DataFrame:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to break on natural boundaries first, only fall back to raw chars.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    rows,n_docs=[],0
    for text,metadata in iter_documents():
        n_docs+=1
        for i,chunk in enumerate(splitter.split_text(text)):
            rows.append({**metadata, "chunk_index": i, "text": chunk})

    df=pd.DataFrame(rows)

    MIN_CHUNK_CHARS = 50
    before = len(df)
    df = df[df["text"].str.strip().str.len() >= MIN_CHUNK_CHARS].reset_index(drop=True)
    logger.info("Dropped %d sub-%d-char junk chunks", before - len(df), MIN_CHUNK_CHARS)



    df.insert(0, "chunk_id", range(len(df)))   # unique integer ID for each chunk
    df.to_parquet(CHUNKS_PATH, index=False)

    logger.info("Chunked %d documents -> %d chunks -> %s", n_docs, len(df), CHUNKS_PATH)
    logger.info("Chunks per source_type:\n%s", df["source_type"].value_counts())
    return df
if __name__ == "__main__":
    df = build_chunks()
    print(df.head())
    print("shape:", df.shape)