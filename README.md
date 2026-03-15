# Bayesian Financial Risk Agent & Market Sentry
# Internship / Capstone Project (AI/ML TRACK)(MAQ SOFTWARE)
A 6-phase quantitative pipeline combining **Bayesian statistics**, **GARCH volatility modelling**, **LSTM machine learning**, and **Retrieval-Augmented Generation (RAG)** to produce daily risk reports with hedging recommendations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  6-PHASE PIPELINE OVERVIEW                          │
├─────────────┬───────────────────────────────────────────────────────┤
│  Phase 1    │  Data Engineering                                     │
│             │  ▸ OHLCV ingestion (yfinance / synthetic)             │
│             │  ▸ Log returns  r_t = ln(P_t / P_{t-1})              │
│             │  ▸ ADF stationarity test                              │
│             │  ▸ Rolling volatility  σ_roll                         │
├─────────────┼───────────────────────────────────────────────────────┤
│  Phase 2    │  Statistical Engine                                   │
│             │  ▸ GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}  │
│             │  ▸ Parametric VaR₉₅ = μ + σ_t · Z₀.₀₅              │
│             │  ▸ Expected Shortfall (CVaR)                          │
│             │  ▸ Volatility regime classification                   │
├─────────────┼───────────────────────────────────────────────────────┤
│  Phase 3    │  ML Predictor                                         │
│             │  ▸ Features: RSI, MACD, Bollinger %B, lagged returns  │
│             │  ▸ 2-layer LSTM binary classifier                     │
│             │  ▸ PyTorch (preferred) / NumPy fallback               │
├─────────────┼───────────────────────────────────────────────────────┤
│  Phase 4    │  RAG System                                           │
│             │  ▸ News ingestion & chunking                          │
│             │  ▸ Embeddings: SentenceTransformer / TF-IDF+SVD      │
│             │  ▸ Vector store: FAISS / NumPy                        │
│             │  ▸ GARCH-weighted retrieval count                      │
├─────────────┼───────────────────────────────────────────────────────┤
│  Phase 5    │  Agent Synthesis                                      │
│             │  ▸ Structured prompt: quant metrics + RAG context     │
│             │  ▸ LLM: Claude (Anthropic) / GPT-4o / Llama3 Ollama  │
│             │  ▸ Rule-based fallback (no API key needed)            │
├─────────────┼───────────────────────────────────────────────────────┤
│  Phase 6    │  Streamlit Dashboard                                  │
│             │  ▸ Price chart + Bollinger Bands                      │
│             │  ▸ Risk gauge (GARCH Vol + VaR breakdown)             │
│             │  ▸ Vol forecast path + return distribution            │
│             │  ▸ RAG news snippets + LLM risk report               │
└─────────────┴───────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Minimal install (no GPU, no API keys):

```bash
pip install numpy pandas scipy statsmodels scikit-learn streamlit plotly
```

### 2. Run the CLI Pipeline

```bash
python main.py --ticker AAPL --years 5 --backend rule
```

With an LLM:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --ticker AAPL --backend anthropic
```

### 3. Launch the Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Mathematical Foundations

### GARCH(1,1)

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

- **ω**: long-run variance constant  
- **α**: ARCH effect (reaction to shocks)  
- **β**: GARCH persistence  
- Stationarity condition: α + β < 1  

### Parametric VaR

```
VaR_{95%} = -(μ·h + σ_t·√h · z_{0.05})
```

where `z_{0.05} ≈ -1.645` (5th percentile of N(0,1)).

### Expected Shortfall (CVaR)

```
ES_{95%} = σ_t · φ(z_{0.05}) / (1 - 0.95)
```

---

## File Structure

```
bayesian_risk_agent/
├── main.py                   # Orchestrator — runs all phases
├── requirements.txt
├── README.md
└── src/
    ├── data_engineering.py   # Phase 1 — data + log returns + ADF
    ├── statistical_engine.py # Phase 2 — GARCH + VaR
    ├── ml_predictor.py       # Phase 3 — features + LSTM
    ├── rag_system.py         # Phase 4 — embeddings + vector store
    ├── agent_synthesis.py    # Phase 5 — LLM prompt + report
    └── dashboard.py          # Phase 6 — Streamlit UI
```

---

## Replacing Synthetic Data with Real Data

Everything works out-of-the-box with synthetic data. To upgrade each data source, just install the package and supply the key — no code changes needed.

---

### 1. Real Market Data  (`data_engineering.py` — `generate_market_data`)

**File to enable:** nothing — it auto-detects `yfinance`.

```bash
pip install yfinance
```

That's it. The function already calls `yf.download()` when `yfinance` is importable and falls back to synthetic data if the download fails. Supported tickers:

| Asset | Ticker |
|---|---|
| Apple | `AAPL` |
| NIFTY 50 | `^NSEI` |
| S&P 500 ETF | `SPY` |
| Tesla | `TSLA` |
| NVIDIA | `NVDA` |
| Bitcoin | `BTC-USD` |

---

### 2. Real News Headlines  (`data_engineering.py` — `generate_news_data`)

**File to enable:** nothing — it auto-detects `newsapi-python` + your key.

```bash
pip install newsapi-python
```

Get a free key at **https://newsapi.org/register** (takes 30 seconds).

```bash
# Pass via environment variable (recommended)
export NEWS_API_KEY=your_key_here
python main.py --ticker AAPL

# Or pass directly on the CLI
python main.py --ticker AAPL --news-api-key your_key_here

# Or paste it in the dashboard sidebar (🔑 API Keys section)
streamlit run src/dashboard.py
```

> ⚠️ **Free tier limit:** articles from the last 30 days only, 100 requests/day.
> For historical news (5-year project), consider upgrading to NewsAPI paid,
> or use the GNews API / Alpha Vantage News as alternatives.

---

### 3. Better Embeddings  (`rag_system.py` — `get_embedder`)

**File to enable:** nothing — it auto-detects `sentence-transformers`.

```bash
pip install sentence-transformers
```

Models download automatically from HuggingFace on first use. Choose via the dashboard **Embedding Model** dropdown or `--embed-model` CLI flag:

| Model | Size | Best For |
|---|---|---|
| `ProsusAI/finbert` | ~440 MB | ★ Financial/news text — most relevant |
| `all-MiniLM-L6-v2` | ~90 MB | Fast general-purpose — good default |
| `all-mpnet-base-v2` | ~420 MB | Highest quality, slower |
| `tfidf` | 0 MB | Always works offline, no download |

---

### One-command real-data setup (all three at once)

```bash
# Install all real-data packages
pip install yfinance newsapi-python sentence-transformers groq

# Set keys
export GROQ_API_KEY=gsk_...         # from console.groq.com (free)
export NEWS_API_KEY=your_key...     # from newsapi.org (free)

# Run with real data + best embeddings + Groq LLM
python main.py \
  --ticker AAPL \
  --backend groq \
  --embed-model all-MiniLM-L6-v2 \
  --news-api-key $NEWS_API_KEY

# Or launch dashboard (paste keys in sidebar)
streamlit run src/dashboard.py
```

The dashboard shows a **live status banner** under the metrics strip telling you exactly which data source is active for each component.

---

## LLM API Keys

| Backend    | Environment Variable    | Model Used                    | Cost  |
|------------|------------------------|-------------------------------|-------|
| **Groq**   | `GROQ_API_KEY`         | llama-3.3-70b-versatile       | **Free tier** |
| Anthropic  | `ANTHROPIC_API_KEY`    | claude-opus-4-5               | Paid  |
| OpenAI     | `OPENAI_API_KEY`       | gpt-4o-mini                   | Paid  |
| Ollama     | `OLLAMA_HOST`          | llama3 (local)                | Free  |
| Rule-based | *(no key needed)*      | Deterministic logic           | Free  |

### Getting a Groq API Key (Free)
1. Go to **https://console.groq.com**
2. Sign up → API Keys → Create key
3. Copy the key (starts with `gsk_...`)

```bash
# Set once in your shell
export GROQ_API_KEY=gsk_...

# CLI — uses Groq automatically
python main.py --ticker AAPL --backend groq

# Choose a faster/lighter model
python main.py --ticker AAPL --backend groq --groq-model llama-3.1-8b-instant

# Dashboard — select "groq" in the sidebar, paste key or set env var
streamlit run src/dashboard.py
```

---

## Extending the Project

- **Hidden Markov Model** for regime detection (replace `classify_regime`)
- **Temporal Fusion Transformer** (TFT) for Phase 3
- **Monte Carlo VaR** via GARCH path simulation
- **Options Greeks** (delta/vega hedges) in the recommendation engine
- **Backtesting** framework (Zipline / Backtrader integration)
- **Multi-asset portfolio** VaR with correlation matrix

---

## Disclaimer

This project is for **educational purposes only**. It is not financial advice.
All risk models have limitations and should not be used as the sole basis for
investment decisions.
