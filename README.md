# 🏆 FinSentinel AI — Autonomous Financial Intelligence Platform

> An end-to-end AI platform that collects SEC filings, stock prices, financial news, and
> social sentiment, analyzes them with ML/DL/NLP models, and answers questions through a
> RAG-powered AI assistant — presented in an interactive dashboard.

**Status:** 🚧 In development (30-day build: Jul 15 → Aug 13, 2026)

<!-- TODO Day 29: replace with real architecture diagram + screenshots + demo GIF -->

## The Problem

Investors must read thousands of pages across 10-K/10-Q filings, earnings calls, news,
and social media before making decisions. FinSentinel AI automates the collection and
analysis of these sources and returns concise, evidence-backed insights.

## Features

- 📊 **Company Dashboard** — stock performance, sentiment trends, risk score, AI summary
- 🤖 **AI Chat Assistant** — grounded (RAG) answers over SEC filings & transcripts
- 📈 **Stock Trend Prediction** — XGBoost + LSTM models on engineered features
- ⚠️ **Financial Risk Scoring** — ML model over structured financial metrics
- 📰 **Sentiment Analysis** — FinBERT over news, Reddit, and earnings calls
- 📝 **Automatic Report Generation** — AI-generated investment summaries

## Architecture

```
Data Sources (SEC EDGAR · yfinance · News · Reddit · Earnings Calls)
        │
Data Collection → Cleaning → Feature Engineering
        │
 ┌──────┴───────┐
 Structured      Text
 ML/DL Models    NLP Pipeline (sentiment · NER)
 └──────┬───────┘
        │
Embeddings (Sentence Transformers) → FAISS
        │
RAG Pipeline (LangChain) → LLM (Ollama local / Groq cloud)
        │
Streamlit Dashboard + AI Chat
```

## Tech Stack

Python · Pandas · scikit-learn · XGBoost · PyTorch · Hugging Face Transformers ·
Sentence Transformers · spaCy · FAISS · LangChain · Ollama / Groq · FastAPI ·
Streamlit · Plotly · SQLite · Docker

## Setup

```bash
git clone <repo-url>
cd FinSentinel
conda create -n finsentinel python=3.12   # or: python -m venv venv
conda activate finsentinel
pip install -r requirements.txt
copy .env.example .env       # then fill in your keys
```

<!-- TODO Day 28: add run instructions (streamlit run app.py) and deployment link -->

## Project Structure

```
FinSentinel/
├── data/            # raw / processed / embeddings (gitignored)
├── notebooks/       # EDA & experiments
├── models/          # trained model artifacts (gitignored)
├── src/
│   ├── data_pipeline/   # collectors: stocks, SEC, news, reddit
│   ├── preprocessing/   # cleaning & feature engineering
│   ├── ml_models/       # XGBoost, risk scoring
│   ├── dl_models/       # LSTM price models (PyTorch)
│   ├── nlp/             # sentiment, NER
│   ├── rag/             # chunking, FAISS, retrieval
│   ├── llm/             # LLM chains, prompts, summarization
│   ├── dashboard/       # Streamlit pages & components
│   └── utils/           # shared helpers
├── mentorship/      # 30-day build plan & progress tracker
├── requirements.txt
└── README.md
```

## Disclaimer

Educational portfolio project. Not financial advice.
