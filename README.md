# Kronos: A Foundation Model for the Language of Financial Markets 

> **Reproduction of Kronos (AAAI 2026) with interactive dashboard, backtesting simulator, sentiment overlay, and KonaDB result persistence.**

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2508.02739-red)](https://arxiv.org/abs/2508.02739)
[![Venue](https://img.shields.io/badge/venue-AAAI%202026-blue)](https://aaai.org)
[![HuggingFace](https://img.shields.io/badge/model-NeoQuasar-yellow)](https://huggingface.co/NeoQuasar)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/konaaravind4/kronos-reproduction?style=social)](https://github.com/konaaravind4/kronos-reproduction)

**Kronos** is the first open-source foundation model for financial candlestick (K-line) data, trained on **12 billion+ K-line records** from **45 global exchanges**. This repo reproduces all paper results, adds an interactive dashboard, a portfolio backtesting simulator, and integrates with the Kona AI Ecosystem for sentiment-enhanced forecasting.

> **Official Repo**: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) · **HuggingFace**: [NeoQuasar](https://huggingface.co/NeoQuasar)

---

##  Key Results (from paper)

| Task | Metric | Kronos vs Best TSFM | Kronos vs Best Non-Pretrained |
|------|--------|--------------------|-----------------------------|
| Price Forecasting | RankIC | **+93%** | **+87%** |
| Volatility Forecasting | MAE | **−9%** | — |
| Synthetic Generation | Fidelity | **+22%** | — |

---

##  Model Architecture

```
Input (OHLCVA, D=6)
        │
        ▼
┌─────────────────────────────────────┐
│  K-line Tokenizer (BSQ)             │
│  Binary Spherical Quantization      │
│  → Coarse tokens + Fine tokens      │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Autoregressive Transformer         │
│  GPT-style decoder-only             │
│  Pre-trained: next-token prediction │
└─────────────────────────────────────┘
        │
        ▼
Tasks: Forecasting · Volatility · Generation
```

---

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Market Data

```bash
python scripts/fetch_data.py
# Downloads OHLCVA for BTC, ETH, SPY, AAPL, MSFT, NVDA, TSLA via yfinance
```

### 3. Run All Experiments

```bash
# With Kronos model (~100MB download from HuggingFace)
python run_all.py

# Simulation mode (no download, calibrated to paper results)
python run_all.py --no-model
```

### 4. View Interactive Dashboard

Open `interface/index.html` in your browser for all results, charts, and a live forecast demo.

---

##  Project Structure

```
kronos-reproduction/
├── interface/
│   └── index.html              # Interactive professor-ready dashboard
├── scripts/
│   ├── fetch_data.py           # Download OHLCVA market data
│   ├── baselines.py            # Naive, MovAvg, ARIMA, GARCH, EWMA
│   ├── experiment_forecasting.py       # Price series forecasting (§4.1)
│   ├── experiment_volatility.py        # Volatility forecasting (§4.2)
│   ├── experiment_generation.py        # Synthetic K-line generation (§4.3)
│   ├── experiment_test_time_scaling.py # Test-time scaling (Fig. 7)
│   └── backtest.py             # 🆕 Portfolio backtesting simulator
├── integrations/
│   ├── sentiment_overlay.py    # 🆕 Pull sentiment from Dashboard
│   └── kona_storage.py         # 🆕 Persist results in KonaDB
├── Kronos/                     # Official repo (git submodule)
├── data/                       # Downloaded market data (auto-created)
├── results/                    # Experiment outputs (auto-created)
├── run_all.py                  # Master script
└── requirements.txt
```

---

##  Backtesting Simulator (New!)

Test Kronos-driven trading strategies against historical data:

```bash
# Run all built-in strategies and compare
python scripts/backtest.py \
  --prices data/btc_prices.json \
  --forecasts results/kronos_forecasts.json \
  --asset BTC \
  --capital 10000 \
  --kona-db results/backtest.kona
```

```
═══════════════════════════════════════════════════════
  Backtest Results — SentimentBlendStrategy
  Asset: BTC
═══════════════════════════════════════════════════════
  Initial Capital : $10,000.00
  Final Capital   : $13,247.82
  Total Return    : +32.48%
  Total Trades    : 47
  Win Rate        : 61.7%
  Max Drawdown    : 8.3%
  Sharpe Ratio    : 1.847
═══════════════════════════════════════════════════════
```

**Strategies available:**

| Strategy | Description |
|----------|-------------|
| `ThresholdStrategy` | Long/short when forecast exceeds ±threshold |
| `MomentumStrategy` | Position size ∝ forecast confidence |
| `SentimentBlendStrategy` | Blends Kronos forecast + market sentiment |

---

##  Sentiment Overlay (New!)

Enhance Kronos forecasts with real-time market sentiment from the [Sentiment Dashboard](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard):

```python
from integrations.sentiment_overlay import SentimentOverlay

overlay = SentimentOverlay(sentiment_api="http://localhost:8000")

# Get current market sentiment
mood = overlay.get_market_mood(ticker="BTC", hours=1)
print(f"Signal: {mood['dominant']} | Score: {mood['avg_score']}")

# Adjust Kronos forecast based on sentiment
adjusted = overlay.adjust_forecast(
    kronos_forecast=44000.0,
    ticker="BTC",
    blend_weight=0.2   # 20% sentiment influence
)
print(f"Raw: 44000.0 → Adjusted: {adjusted:.2f}")
```

---

##  KonaDB Integration (New!)

Persist all experiment results in [KonaDB](https://github.com/konaaravind4/kona-db) for cross-experiment comparison:

```python
from integrations.kona_storage import KronosStorage

storage = KronosStorage("results/experiments.kona")

# Store experiment result
storage.save_experiment(
    experiment="forecasting",
    asset="BTC",
    metric="RankIC",
    value=0.127,
    model="kronos",
    notes="baseline comparison §4.1"
)

# Compare experiments
storage.compare(experiment="forecasting", metric="RankIC")
```

```bash
# Query results in plain English with AI SQL Analyst
curl -X POST http://ai-sql:8000/query \
  -d '{"question": "Which experiment had the highest RankIC for BTC?", "db_path": "results/experiments.kona"}'
```

---

##  Ecosystem Integration

```
Kronos Reproduction
     │
     ├── Backtesting ──────────────────────────────► Portfolio analysis
     │
     ├── Sentiment Overlay ────────────────────────► Real-time-Sentiment-Intelligence-Dashboard
     │   (bullish/bearish market signals)             (financial sentiment mode)
     │
     ├── KonaDB Results Storage ───────────────────► kona-db
     │   (OHLCVA time-series + experiment results)    (time-series + vector store)
     │
     └── Knowledge Retrieval ──────────────────────► RAG-GraphRAG-Knowledge-Engine
         (financial papers, quant strategies)         (financial namespace)
```

---

##  Related Projects

| Project | Integration |
|---------|-------------|
| [Real-time-Sentiment-Intelligence-Dashboard](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard) | Market sentiment overlay on forecasts |
| [kona-db](https://github.com/konaaravind4/kona-db) | Persists OHLCVA and experiment results |
| [AI-SQL-Data-Analyst](https://github.com/konaaravind4/AI-SQL-Data-Analyst) | Query experiment results in natural language |
| [RAG-GraphRAG-Knowledge-Engine](https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine) | Financial knowledge retrieval |

---

##  License

MIT © [konaaravind4](https://github.com/konaaravind4)
