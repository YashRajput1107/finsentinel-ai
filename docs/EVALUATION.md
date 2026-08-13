# Evaluation Report

This document records how the models in this project were evaluated, what the results were,
and what those results mean. It includes a null result and a rejected hypothesis, both
reported as they occurred.

## Contents

1. Evaluation methodology
2. Stock direction models (2021–2026 data)
3. Investigation: is ~60% directional accuracy achievable?
4. Experiment: does sentiment add predictive signal?
5. Sentiment classification results
6. Financial risk score
7. RAG system evaluation
8. Limitations
9. Conclusions and future work

---

## 1. Evaluation methodology

The same protocol is applied throughout:

- **Chronological splits only.** Time-series data is never shuffled. Training data always
  precedes test data, so the model is evaluated on a period it has not seen.
- **A naive baseline is reported alongside every accuracy figure.** For directional
  prediction the baseline is "always predict up," which is non-trivial because equity
  markets drift upward. An accuracy figure without its baseline is not interpretable.
- **Walk-forward validation** (`TimeSeriesSplit`) is used to check whether a result holds
  across multiple time windows rather than one potentially lucky split.
- **Leakage controls.** All features are computed from backward-looking windows only; the
  target is the only forward-looking column; scalers are fit inside a pipeline on training
  data only. A per-feature leakage checklist is maintained in `notebooks/08_features.ipynb`.
- **Metric choice matches the task.** Macro-F1 is used for the imbalanced sentiment task;
  accuracy, precision, recall, F1, ROC-AUC and PR-AUC are reported for directional models.

---

## 2. Stock direction models (2021–2026 data)

Features: lagged returns (1/2/3/5/10-day), price-to-moving-average ratios, 20-day rolling
volatility, RSI(14), MACD, volume z-score. Target: next-day direction. Split: train through
2023, test from 2025.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Always-up baseline | 0.531 | — |
| Logistic Regression | 0.529 | 0.505 |
| Random Forest | 0.523 | 0.502 |
| XGBoost | 0.510 | 0.510 |
| LSTM (60-day sequences) | 0.470 | — |

Walk-forward (5 folds, pre-2025): Logistic Regression 0.502 ± 0.019, XGBoost 0.501 ± 0.014.

**Result: no model beats the naive baseline, and ROC-AUC is approximately 0.50** — meaning
the models have no ability to rank up-days above down-days. This is consistent across three
model families (linear, tree ensemble, recurrent neural network).

The LSTM performed worst. Its training loss decreased while validation loss increased,
which is the signature of overfitting: with no learnable signal available, the additional
capacity of a sequence model was spent memorising noise.

Logistic Regression was selected as the shipped model. It tied XGBoost within the noise
margin on walk-forward validation, so the simpler model was preferred.

---

## 3. Investigation: is ~60% directional accuracy achievable?

### Framing

This was treated as a hypothesis to be tested, not a target to optimise towards. Repeatedly
adjusting models or features until a test set reaches a chosen number overfits the research
process itself and produces a figure that will not generalise.

### Pre-registered hypothesis (recorded before running the experiment)

> Given a baseline near 0.53, ROC-AUC near 0.50 on price-only features, and the noise
> inherent in next-day price direction, reaching approximately 60% out-of-sample
> directional accuracy appears unlikely. This is stated as a hypothesis to be tested.

**Success criterion, fixed in advance:** an added information source counts as an
improvement only if it exceeds the fold-to-fold noise level (±0.02 accuracy), not by any
small positive difference.

### Data audit (performed first)

| Dataset | Coverage |
|---|---|
| Price features (original) | 2021-09-29 → 2026-07-13 |
| Financial news | 2011-03-03 → 2020-06-11 |
| Reddit posts | 2021-01-28 → 2021-08-16 |

**Finding: the datasets did not overlap at all.** The news data ends 475 days before the
price features begin. The intended experiment — adding sentiment features to the price
model — was therefore impossible with the existing data, and a naive merge would have
produced an empty or silently misaligned dataset.

The cause was a difference in how each source was anchored in time: prices were fetched as
"the last five years" relative to the current date, while the news dataset is a static
snapshot ending in 2020.

**Resolution:** price history was re-fetched for the news period (2011-01 → 2020-06,
starting early to allow for the 50-day feature warm-up) and features were recomputed using
the existing `compute_features()` function without modification. This produced 22,993
feature rows overlapping the news data by 3,376 days. The original 2021–2026 feature set
was left untouched.

---

## 4. Experiment: does sentiment add predictive signal?

### Design

- **Model A:** price features only (11 features).
- **Model B:** price features + sentiment features (13 features).
- Identical model class (Logistic Regression in a scaling pipeline), identical rows,
  identical splits. The only difference between the two is the two additional columns.

**Sentiment features:** FinBERT labels were converted to a signed value
(positive = +1, negative = −1, neutral = 0) weighted by model confidence, then aggregated
per ticker per day into a mean sentiment and a news count.

**Leakage control specific to this experiment:** sentiment features were lagged by one day.
Because the target is the return from today's close to tomorrow's close, using same-day
news risks including items published after the close, which would not have been available
at prediction time.

### Protocol amendment (made after the data audit, before any results)

News coverage is 14.8% of ticker-days overall, and highly uneven (MSFT 0%, AMZN 1.2%,
AAPL 2.5%, JNJ 46.5%, NVDA 40.7%). Filling the remaining 85% of rows with a constant would
confound the effect of sentiment with the effect of missingness. The experiment was
therefore restricted to rows where sentiment exists (n = 3,405), which sharpens the
question to: *on days when news exists, does sentiment add signal beyond price features?*

This amendment was based on data availability, not on any observed result.

### Single-window result (chronological 80/20 split)

Test window: 2019-12-27 → 2020-06-12 (681 rows). Baseline 0.508.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| A: price only | 0.5419 | 0.5457 | 0.5867 | 0.5655 | 0.5709 | 0.5641 |
| B: price + sentiment | 0.5580 | 0.5581 | 0.6243 | 0.5894 | 0.5778 | 0.5713 |

Difference: +0.0162 accuracy, +0.0069 ROC-AUC.

This appeared promising, but **it did not meet the pre-registered success criterion of
±0.02**, and the test window contains the COVID-19 market crash — an unrepresentative
regime. It was therefore treated as provisional and subjected to a robustness check.

### Robustness check (walk-forward, 5 folds)

| Fold | Test start | n | Baseline | Acc A | Acc B | Delta |
|---|---|---|---|---|---|---|
| 1 | 2014-05-12 | 567 | 0.549 | 0.5009 | 0.4868 | −0.0141 |
| 2 | 2017-05-03 | 567 | 0.524 | 0.5379 | 0.5485 | +0.0106 |
| 3 | 2018-11-15 | 567 | 0.519 | 0.4956 | 0.4780 | −0.0176 |
| 4 | 2019-07-26 | 567 | 0.566 | 0.5115 | 0.5309 | +0.0194 |
| 5 | 2020-02-06 | 567 | 0.501 | 0.5503 | 0.5503 | 0.0000 |

- Mean difference: **−0.0003 ± 0.0158**
- Folds where B outperformed A: **2 of 5**
- Mean accuracy: A 0.5192, B 0.5189, baseline 0.5316
- Mean ROC-AUC: A 0.519, B 0.525

### Conclusion of the experiment

**Sentiment, as available in this dataset, does not add next-day directional signal beyond
price features.** The apparent single-window improvement did not survive evaluation across
multiple periods; it is consistent with noise. Both models also fail to beat the naive
baseline on average across folds.

**Conclusion on the 60% question: the hypothesis is rejected.** Across three model families,
two feature sets, and multiple validation windows, out-of-sample directional accuracy
remains near the baseline. There is no evidence that ~60% is achievable with this data and
target, and a reported figure at that level would warrant scrutiny for leakage or an
unrepresentative test set.

### Why the experiment may have failed (hypotheses, not excuses)

- **Sparse coverage.** Sentiment exists for only 14.8% of ticker-days, and near-zero for
  several large tickers.
- **Aggregation may be too coarse.** Markets react to news within minutes; daily averaging
  may destroy the effect.
- **Headlines only.** The dataset contains headlines, not article bodies.
- **The target may be wrong.** Next-day direction is close to unpredictable by construction.
  Volatility, which exhibits clustering, is a more tractable target.

---

## 5. Sentiment classification results

Financial PhraseBank (75% agreement tier), stratified split, 690 test sentences.

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | 0.752 | 0.814 |
| FinBERT (`ProsusAI/finbert`) | **0.938** | 0.946 |

Per-class (FinBERT): negative F1 0.932, neutral 0.960, positive 0.923. Negative-class recall
is 0.976, which matters because negative sentiment is both the rarest class (12%) and the
most operationally important.

Macro-F1 is reported rather than accuracy because 62% of sentences are neutral; a classifier
that always predicted "neutral" would score 62% accuracy while providing no information.

FinBERT was used in inference only — we did not fine-tune it.

### Important caveat — benchmark contamination

The reported 0.938 should be read as an **optimistic upper bound, not real-world
performance**, for two reasons:

1. **Evaluated on its own training data.** ProsusAI's FinBERT was itself fine-tuned on the
   Financial PhraseBank — the same dataset used here to evaluate it. Although a train/test
   split was made, FinBERT had already seen these sentences during *its* training, so the
   test split is not genuinely held out for it. This is train/test contamination at the
   level of the benchmark's provenance, and it inflates the score. It also makes the
   comparison against the TF-IDF baseline (which was trained only on the local train split)
   not strictly apples-to-apples — FinBERT effectively had a home-field advantage.
2. **Distribution shift, unmeasured.** The 0.938 is on curated, single-sentence PhraseBank
   text. FinBERT was then applied to the project's actual news headlines and Reddit posts,
   which are a different, messier distribution — and unlabeled, so its accuracy on that real
   target text is unknown.

A fair assessment would evaluate both models on a *separate* financial-sentiment dataset that
neither had seen (noted as future work). The honest claim is therefore: FinBERT achieves
0.938 macro-F1 on the PhraseBank benchmark, which is its own training distribution; its
performance on unseen, in-the-wild financial text is not established here.

**Note on interpretation:** strong sentiment classification does **not** imply a strong
market predictor. Section 4 shows this directly — accurate sentiment labels did not
translate into directional predictive power. These are two distinct problems: one is a text
classification task with a learnable pattern; the other is a forecasting task against an
adaptive market.

---

## 6. Financial risk score

A transparent 0–10 score summarising each company's balance-sheet risk. It is
**descriptive, not predictive** — there is no labeled risk target, so a documented
weighted rule is used rather than an ML model (which would be an unvalidated black box).

**Factors** (min-max normalised within the ten-company universe, oriented so higher =
riskier), with weights: leverage / debt-to-equity (0.20), illiquidity / inverse current
ratio (0.20), low profitability / inverse profit margin (0.20), beta (0.15), volatility
(0.25).

**Result** (low → high risk): GOOGL 2.85, NVDA 3.55, JPM 4.41, MSFT 4.49, XOM 4.52,
JNJ 4.81, META 5.54, AAPL 6.50, AMZN 6.64, TSLA 7.26.

**Interpretation and known limitations:**

- It measures **financial** risk (balance-sheet health), **not price risk**. NVDA ranks
  low-risk despite being highly volatile because its fundamentals are strong (high margins,
  low debt). Price volatility is therefore reported separately, and the metric is labeled
  "Financial Risk Score" rather than "Risk Score" to avoid misleading a user.
- **No sector adjustment:** a bank (JPM) carries structurally high leverage that is not
  normalised against its sector.
- **Ratio blind spots:** AAPL and AMZN rank high because a ~1.0 current ratio and thin
  margins do not capture Apple's cash reserves or Amazon's deliberate low-margin strategy.
- **Survivorship bias** applies (all ten firms are currently healthy), so the score range
  is compressed and ranks relative risk among survivors only.

## 7. RAG system evaluation

The retrieval-augmented generation (RAG) pipeline answers natural-language questions over
SEC filings (10-K/10-Q) and earnings-call transcripts for the ten companies. Pipeline:
chunk → embed (all-MiniLM-L6-v2) → FAISS cosine index → metadata-filtered retrieval with
MMR diversity → grounded answer via a local LLM (Ollama, `llama3.2`) instructed to answer
only from retrieved context and to refuse otherwise.

### Retrieval evaluation (Day 20)

A 12-question gold set (one per company topic) was used to measure whether the correct
company's chunks are retrieved, without pre-filtering by ticker.

| Metric | Result |
|---|---|
| hits@1 | 11/12 (0.92) |
| hits@3 | 12/12 (1.00) |
| hits@5 | 12/12 (1.00) |

**Read with caveats:** each question names its company (an easy retrieval signal),
ticker-match is a proxy for true relevance, and n = 12 is a smoke test, not a benchmark.

### Answer-pipeline evaluation (Day 23)

A 15-question set (12 answerable + 3 deliberately unanswerable) tested the full
retrieve → ground → cite → refuse pipeline. The three unanswerable questions were designed
to be *tempting* — retrieval returns topically-related chunks — so they test grounding under
pressure, not trivial empty-retrieval refusal (a company outside the universe; a
forward-looking prediction; a granular datum absent from filings).

| Metric | Result | Interpretation |
|---|---|---|
| Refusal accuracy (unanswerable) | 3/3 | Correctly refused all tempting out-of-corpus questions |
| Citation rate (answerable) | 9/12 clean | Non-citing answers are the false refusals |
| Groundedness (answerable, answered) | 9/9 | Every answered claim traces to a cited chunk (author-judged; one numeric claim — Apple "45.5m iPhones" — verified against the source) |
| Answer relevance (answerable, answered) | 9/9 | Every answer addresses the question asked |
| **False-refusal rate (answerable)** | **3/12 (0.25)** | Refused three questions it had the context to answer |

**Groundedness and relevance were judged manually**, not by an LLM judge: the local judge
model (`llama3.2`, 3B) is weaker than the generator, so a weak auto-grader would add noise,
not rigor. At n = 15, careful human judgement is the more honest instrument.

### The false-refusal finding (investigated, not hidden)

The 25% false-refusal rate is the pipeline's real weakness. Investigation:

- **MMR trade-off.** "Where does JPMorgan get most of its revenue?" refuses *stably* (5/5 runs)
  only when MMR is enabled. Diversifying the context trades away some of the single
  most-answer-bearing chunk, and the strict 3B model then cannot locate a crisp answer.
  MMR reduced redundancy (Section on Day 22: worst-pair similarity 1.0 → 0.63) but slightly
  raised false refusals — a genuine precision/recall trade-off.
- **Small-model instability.** Other borderline questions (Tesla, Alphabet) flip between
  answer and refusal across runs, so `temperature = 0` is only approximately deterministic
  for this model.

**Framing.** A 25% false-refusal rate is high for usability, but over-refusal is the safe
failure direction for a financial tool — refusing is preferable to hallucinating. No answer
in the evaluation was found to be ungrounded. Reducing false refusals (higher MMR λ, a larger
local model such as `llama3.1:8b`, or a larger context `k`) is a deployment-tuning task, and
the number is reported rather than tuned away.

### Re-evaluation after the inference-provider switch

The deployed application cannot run a local model (Streamlit Community Cloud provides no GPU
and no background LLM server), so generation was moved behind a provider abstraction: Ollama
`llama3.2` (3B) for local development, Groq `llama-3.3-70b-versatile` (70B) for deployment.

**A provider switch is a model change, so the evaluation was re-run rather than assumed to
carry over.** The same 15-question set, same retrieval, same prompt — only the generator differs:

| Metric | Ollama `llama3.2` (3B) | Groq `llama-3.3-70b` (70B) |
|---|---|---|
| Refusal accuracy (unanswerable) | 3/3 | **3/3** |
| Citation rate (answerable) | 9/12 | **11/12** |
| False refusals (answerable) | 3/12 (0.25) | **1/12 (0.08)** |

**Findings.**

- **The guardrail held.** The larger model still refused all three tempting out-of-corpus
  questions. This was the main risk of the switch: a stronger model has more world knowledge
  to fall back on and could have answered from memory instead of admitting the documents do
  not cover the question. It did not.
- **False refusals fell from 25% to 8%.** Only the Meta competition question is still refused.
  This supports the earlier diagnosis that the false refusals came largely from a small model
  failing to locate a crisp answer in a diversified context, rather than from a retrieval fault.
- **Answers are more specific and better cited.** The JPMorgan revenue question, which the 3B
  model refused in 5 of 5 runs, is now answered with figures traced to the source: total net
  revenue of $46.4 billion of which $24.0 billion was net interest income — both verified
  verbatim in the cited chunk (JPM 10-Q, 2025-11-04).
- Groundedness and relevance remain author-judged; no ungrounded claim was found in the
  answered set, and the numeric claim above was checked against its source chunk.

**Caveat unchanged:** n = 15 is a smoke test, not a benchmark. The improvement is consistent
across all three metrics but rests on a small sample.

---

## 8. Limitations

- **Temporal mismatch between sources.** News (2011–2020), Reddit (2021), and the original
  price features (2021–2026) do not align. The sentiment experiment was only possible after
  re-deriving price history for the news period.
- **Sparse news coverage.** 14.8% of ticker-days overall; MSFT has none.
- **Survivorship bias.** The ten companies are large, currently-successful firms selected
  with hindsight. Companies that failed during the period are absent, which likely biases
  results optimistically.
- **No transaction costs modelled.** Directional accuracy is not the same as profitability;
  a cost-aware backtest is not yet implemented.
- **Single asset class, single market.** US large-cap equities only.
- **Static data.** No live feed; all sources are historical snapshots.
- **RAG corpus gaps.** Earnings-call transcripts exist for only 5 of the 10 companies
  (AAPL, AMZN, GOOGL, MSFT, NVDA); all 10 have 10-K/10-Q filings. Source text also carries
  minor encoding artifacts from extraction.
- **RAG false refusals.** The answer pipeline refuses some answerable questions (safe
  direction, but a usability cost): ~25% with the local 3B model, falling to ~8% with the
  70B deployment model. Driven by an MMR precision/recall trade-off and generator capacity.
- **Two generators, one evaluation set.** Local development and deployment run different
  models, so results are reported per provider (Section 7) rather than as a single number.

---

## 9. Conclusions and future work

1. Next-day directional prediction from public daily price features produces a null result.
   This is the expected outcome and is reported rather than concealed.
2. Adding news sentiment did not change that, when evaluated on the period where both data
   sources exist and across multiple time windows.
3. Financial sentiment classification is a genuinely learnable task; FinBERT achieves
   0.938 macro-F1.
4. The most informative part of this exercise was the methodology: an apparent improvement
   from a single test window was shown to be noise once evaluated across five windows.
5. The RAG pipeline grounds and cites answers over the filing/transcript corpus and refuses
   tempting out-of-corpus questions (3/3 on both generators tested). Its main weakness is a
   false-refusal rate on answerable questions — 25% with the local 3B model, 8% with the 70B
   deployment model — a documented precision/recall trade-off, reported rather than tuned away.
6. Switching inference providers was treated as a model change: the evaluation set was re-run
   on the new generator rather than assuming the previous numbers still applied.

**Future work, in order of expected value:**

- Change the target from direction to **volatility**, which exhibits clustering and is more
  predictable.
- Obtain **temporally aligned, denser news data** covering the same period as the price data.
- Test **intraday** rather than daily sentiment aggregation.
- Add a **cost-aware backtest** (transaction costs, slippage, Sharpe ratio, maximum
  drawdown) so that model quality is assessed economically, not only statistically.
- **Re-evaluate sentiment on a held-out dataset** that FinBERT was not trained on, to
  measure its true generalization rather than its performance on its own training
  distribution (see the benchmark-contamination caveat in Section 5).
- **Reduce RAG false refusals** by tuning MMR λ toward relevance, testing a larger local
  generator (`llama3.1:8b`), or increasing retrieved context `k` — measured against the
  15-question answer eval so the guardrail is not weakened in the process.
