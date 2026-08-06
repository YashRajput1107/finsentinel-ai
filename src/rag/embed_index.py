import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "rag_chunks.parquet"
INDEX_DIR   = PROCESSED_DIR / "rag_index"
INDEX_PATH  = INDEX_DIR / "chunks.faiss"       # the FAISS vectors
META_PATH   = INDEX_DIR / "chunks_meta.parquet"  # metadata, SAME order as vectors

EMBED_MODEL = "all-MiniLM-L6-v2"

def build_index() -> None:
    # reset_index(drop=True) LOCKS the order now. Everything downstream trusts this order.
    df = pd.read_parquet(CHUNKS_PATH).reset_index(drop=True)
    logger.info("Loaded %d chunks", len(df))

    model = SentenceTransformer(EMBED_MODEL)

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit length -> dot product == cosine similarity
        convert_to_numpy=True,
    ).astype("float32")              # FAISS requires float32
    logger.info("Embeddings shape: %s", embeddings.shape)   # expect (40219, 384)

    dim=embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity for normalized vectors
    index.add(embeddings)
    logger.info("FAISS index: %d vectors, dim %d", index.ntotal, dim)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    df.to_parquet(META_PATH, index=False)   # saved in the SAME order -> the alignment contract
    logger.info("Saved index -> %s | metadata -> %s", INDEX_PATH, META_PATH)

def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    return index, meta


def search(query: str, k: int = 5, ticker=None, source_type=None,
           model: SentenceTransformer | None = None) -> pd.DataFrame:
    model = model or SentenceTransformer(EMBED_MODEL)
    index, meta = load_index()

    q = model.encode([query], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")

    selector, _ids = _build_selector(meta, ticker, source_type)   # _ids kept alive
    if selector is not None:
        params = faiss.SearchParameters(sel=selector)
        scores, idx = index.search(q, k, params=params)   # FAISS only looks at allowed rows
    else:
        scores, idx = index.search(q, k)                  # unfiltered, as before

    idx, scores = idx[0], scores[0]
    idx = idx[idx != -1]          # FAISS returns -1 if fewer than k matches exist
    hits = meta.iloc[idx].copy()
    hits["score"] = scores[:len(idx)]
    return hits[["score", "ticker", "source_type", "form_type", "doc_date", "chunk_id", "text"]]

def _build_selector(meta:pd.DataFrame, ticker=None, source_type=None):
    """Return a FAISS IDSelector limiting search to rows matching the filters,
    or None if no filter is requested. Row position == FAISS id (alignment contract)."""
    if ticker is None and source_type is None:
        return None,None
    mask=np.ones(len(meta),dtype=bool)
    if ticker is not None:
        mask &= (meta["ticker"].values == ticker)
    if source_type is not None:
        mask &= (meta["source_type"].values == source_type)
    ids = np.where(mask)[0].astype("int64")          # allowed row positions
    selector = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
    return selector, ids   # return ids too — keep it alive during the search


    



if __name__ == "__main__":
    build_index()

    res = search("What are the main risks the company faces?", k=5)
    for _, r in res.iterrows():
        print(f"[{r.score:.3f}] {r.ticker} {r.form_type} {r.doc_date} #{r.chunk_id}")
        print("   ", r.text[:160].replace("\n", " "), "...\n")