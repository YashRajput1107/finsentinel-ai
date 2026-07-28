# FinSentinel

An end-to-end financial analysis project that collects data on a fixed set of companies
(stock prices, SEC filings, financial news, Reddit discussion, and earnings-call
transcripts), builds machine-learning and NLP models over that data, and is intended to
surface the results through a retrieval-augmented chatbot and a dashboard.

The project is being built deliberately in phases — data, then models, then testing, then
deployment — and is documented honestly, including the parts that did **not** work. In
particular, the stock-direction models produce a null result (they do not beat a naive
baseline), which is the expected outcome for daily price prediction from public data and
is reported as such rather than hidden.

**Status: in active development.** The data pipeline and the ML/NLP models are built and
evaluated. The RAG chatbot, dashboard, and deployment are not yet implemented. See the
status table below for the current state of each component.

## Motivation

Analysing a single company means reading across many scattered, high-volume sources:
annual and quarterly filings, earnings calls, news, and social media. The goal of this
project is to collect those sources into one reproducible pipeline and apply ML and NLP to
them, while treating the project primarily as an exercise in doing each step *correctly and
defensibly* — leakage-free evaluation, honest metrics, and a clear rationale for every
technical choice.

## Current status

| Component | Status |
|---|---|
| Data pipeline (6 sources, one-command rebuild) | Done |
| Exploratory data analysis | Done |
| Feature engineering (price features: returns, RSI, MACD, volatility, volume) | Done |
| Stock-direction models (Logistic Regression, Random Forest, XGBoost, LSTM) | Done — null result (see Results) |
| Sentiment analysis (TF-IDF baseline + FinBERT) | Done |
| Sentiment scoring of news + Reddit | Done |
| Named-entity recognition | Not implemented |
| Rule-based company risk score | Not implemented |
| RAG chatbot (chunking, embeddings, FAISS, retrieval, grounded answers) | Not implemented |
| Testing suite (`tests/`) | Not implemented |
| Dashboard (Streamlit) | Not implemented |
| Deployment (Docker, hosting) | Not implemented |

## Architecture (target)

The diagram below is the intended end-state. Components marked *(not implemented)* above
do not exist yet.

```
Data sources: yfinance, SEC EDGAR, financial news, Reddit, earnings transcripts
      |
Collectors (src/data_pipeline) -> cleaning -> data/processed (Parquet + SQLite)
      |
  +---+-------------------+
  |                       |
Numeric features       Text
(src/preprocessing)    (src/nlp: FinBERT sentiment)
  |                       |
ML/DL models           [planned] embeddings -> FAISS -> retrieval -> LLM (RAG)
(src/ml_models,
 src/dl_models)
  |                       |
  +----------+------------+
             |
      [planned] Streamlit dashboard + chat
```

## Data sources

| Source | Contents | Coverage | Notes |
|---|---|---|---|
| yfinance | Daily OHLCV | 10 tickers, ~5 years to 2026 | Adjusted prices |
| SEC EDGAR | 10-K / 10-Q filings | 10 tickers, 60 filings | Parsed to plain text |
| Financial PhraseBank | Labeled sentiment sentences | 3,448 (75% agreement tier) | Sentiment training data |
| Financial news | Headlines with ticker + date | 14,303 rows, 9 tickers | **2011–2020 only** |
| Reddit (r/wallstreetbets) | Posts with score + date | 4,072 tagged rows | **Jan–Aug 2021 only** |
| Earnings transcripts | Call transcripts | 95 files, 5 companies | 2016–2020 |

Companies: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, JNJ, XOM.

### Known data limitations

These matter for interpreting the results and are not hidden:

- **Temporal mismatch.** The news data ends in 2020 and the Reddit data covers only 2021,
  while the price data extends to 2026. Sentiment therefore cannot be used as a predictive
  feature for the 2025 test period without temporally aligned data. Any experiment combining
  sentiment with price signals must be run on the period where both sources overlap.
- **No Microsoft news.** The news dataset contains zero MSFT rows; that source simply never
  collected them.
- **Survivorship bias.** The 10 companies are large, currently-successful firms chosen with
  hindsight. A production system would need to include companies that later failed.
- **Historical, not live.** All sources are static snapshots; there is no live data feed.

## Methodology and models

### Feature engineering
Price features are computed per ticker in `src/preprocessing/features.py`: lagged returns
(1/2/3/5/10-day), price-to-moving-average ratios, 20-day rolling volatility, RSI(14), MACD,
and a volume z-score. RSI and MACD are implemented from their formulas rather than a library.
Targets are next-day and 5-day directional labels. All features look only backward; a
written leakage checklist accompanies the notebook.

### Stock-direction models
Logistic Regression, Random Forest, and XGBoost are trained on pooled features across all
tickers, with a chronological split (train through 2023, test from 2025). An LSTM is trained
on 60-day sequence windows. Models are compared against an always-up baseline and validated
with walk-forward (`TimeSeriesSplit`).

### Sentiment
A TF-IDF + Logistic Regression baseline is compared against FinBERT (`ProsusAI/finbert`) run
in inference mode. FinBERT is then used to score all news and Reddit text.

## Results

Stock direction (test set, 2025; always-up baseline = 0.531 accuracy):

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Always-up baseline | 0.531 | — |
| Logistic Regression | 0.529 | 0.505 |
| Random Forest | 0.523 | 0.502 |
| XGBoost | 0.510 | 0.510 |
| LSTM | 0.470 | — |

Walk-forward (5 folds, pre-2025): Logistic Regression 0.502 ± 0.019, XGBoost 0.501 ± 0.014.

**Interpretation:** no model beats the naive baseline, and ROC-AUC is ~0.50 (no ranking
ability). This is the expected result for next-day direction from public price features —
markets arbitrage away obvious signals — and the leakage-free evaluation is what makes the
null result trustworthy rather than a bug. The LSTM did worse because, with no signal to
learn, its extra capacity overfit the training data (training loss fell while validation
loss rose).

Sentiment (Financial PhraseBank test set):

| Model | Macro-F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.752 |
| FinBERT | 0.938 |

FinBERT substantially outperforms the word-counting baseline because it understands context
and finance-specific language (e.g. "costs declined" as positive). This is reported with
macro-F1 rather than accuracy because the classes are imbalanced (~62% neutral).

An investigation into whether adding sentiment features improves stock prediction — run only
on the temporally valid overlap period — is documented in `docs/EVALUATION.md`.

## Evaluation methodology

- **Chronological splits**, never random, for all time-series models, to prevent look-ahead
  leakage.
- **Always-up baseline** reported alongside every accuracy number.
- **Walk-forward validation** to confirm results are stable across time, not a single lucky
  split.
- **Macro-F1** for the imbalanced sentiment task.
- A **leakage checklist** auditing every feature.

## Installation

```bash
git clone <repo-url>
cd FinSentinel
conda create -n finsentinel python=3.12
conda activate finsentinel
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in the values:

```bash
cp .env.example .env
```

- `SEC_EDGAR_USER_AGENT` — required by SEC EDGAR; any real name and email.
- Kaggle API token at `~/.kaggle/kaggle.json` — required for the news and Reddit datasets.
- `GROQ_API_KEY` — not needed yet; planned for the deployed LLM backend.

## Running

Rebuild the full data layer (downloads and processes all six sources):

```bash
python -m src.data_pipeline.run_all
```

Individual stages can be run on their own, e.g. `python -m src.data_pipeline.stock_data`.
Model training and sentiment scoring have their own entry points
(`python -m src.ml_models.trend_classifier`, `python -m src.nlp.sentiment`).
The exploratory and modeling notebooks are in `notebooks/`.

There is no application entry point yet; the dashboard is not implemented.

## Project structure

```
FinSentinel/
├── data/                 raw / processed / embeddings (gitignored)
├── notebooks/            EDA and modeling notebooks (01-12)
├── models/               trained artifacts (gitignored)
├── src/
│   ├── data_pipeline/    collectors: stock_data, sec_filings, text_datasets,
│   │                     reddit_data, transcripts, run_all
│   ├── preprocessing/    features.py (price feature engineering)
│   ├── ml_models/        trend_classifier.py (Logistic Regression)
│   ├── dl_models/        lstm_model.py
│   ├── nlp/              sentiment.py (FinBERT)
│   ├── rag/              (planned)
│   ├── llm/              (planned)
│   ├── dashboard/        (planned)
│   └── utils/            config.py
├── docs/                 EVALUATION.md and other reports
├── requirements.txt
└── README.md
```

## Testing

Not implemented yet. A `tests/` suite (data validation, model/leakage checks, and — once the
RAG component exists — retrieval and grounding checks) is planned before any deployment.

## Deployment

Not implemented yet. Deployment is intentionally deferred until the data, models, and tests
are stable.

## Disclaimer

Educational project. Not investment advice.
