import logging
from functools import lru_cache

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

@lru_cache(maxsize=1)
def load_index():
    """Read the index + aligned metadata once per process. Without the cache this
    re-read ~70MB from disk on EVERY query - cheap on an SSD, expensive on a
    constrained host. lru_cache keeps this module free of any UI framework."""
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    logger.info("index loaded | %d vectors", index.ntotal)
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

def _reconstruct(index, ids):
    """Fetch the stored (normalized) vectors for the given row ids from the FAISS index."""
    return np.vstack([index.reconstruct(int(i)) for i in ids]).astype("float32")


def _mmr(rel, sim, k, lambda_):
    """Maximal Marginal Relevance selection.
    rel : (n,)   relevance of each candidate to the query (cosine)
    sim : (n, n) candidate-to-candidate cosine similarity
    Returns the chosen candidate indices (into the pool), length <= k.
    """
    n=len(rel)
    k = min(k, n)
    selected = [int(np.argmax(rel))]   # start with the most relevant
    while len(selected) < k:
        best,best_score=None,-np.inf
        for c in range(n):
            if c in selected:          # skip ones we've already picked (had 'is' here, always false)
                continue
            redundancy=max(sim[c,s] for s in selected)
            score = lambda_ * rel[c] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best,best_score=c,score
        selected.append(best)
    return selected

def search(query: str, k: int = 5, ticker=None, source_type=None,
           model: SentenceTransformer | None = None,
           use_mmr: bool = False, lambda_: float = 0.6, fetch_k: int = 20) -> pd.DataFrame:
    # bail out early on bad input, no point loading the model for these
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    model = model or SentenceTransformer(EMBED_MODEL)
    index, meta = load_index()

    q = model.encode([query], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")

    selector, _ids = _build_selector(meta, ticker, source_type)
    n_fetch = max(fetch_k, k) if use_mmr else k          # over-fetch a pool for MMR to work with
    if selector is not None:
        scores, idx = index.search(q, n_fetch, params=faiss.SearchParameters(sel=selector))
    else:
        scores, idx = index.search(q, n_fetch)

    idx, scores = idx[0], scores[0]
    keep = idx != -1
    idx, scores = idx[keep], scores[keep]

    if use_mmr and len(idx) > k:
        cand = _reconstruct(index, idx)      # (pool, 384) stored vectors
        sim = cand @ cand.T                  # candidate-to-candidate cosine (normalized -> dot = cosine)
        order = _mmr(scores, sim, k, lambda_)  # scores ARE cosine-to-query (relevance)
        idx, scores = idx[order], scores[order]
    else:
        idx, scores = idx[:k], scores[:k]

    hits = meta.iloc[idx].copy()
    hits["score"] = scores
    return hits[["score", "ticker", "source_type", "form_type", "doc_date", "chunk_id", "text"]]
    
    

if __name__ == "__main__":
    build_index()

    res = search("What are the main risks the company faces?", k=5)
    for _, r in res.iterrows():
        print(f"[{r.score:.3f}] {r.ticker} {r.form_type} {r.doc_date} #{r.chunk_id}")
        print("   ", r.text[:160].replace("\n", " "), "...\n")