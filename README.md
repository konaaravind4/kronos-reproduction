# Kronos: A Foundation Model for the Language of Financial Markets

> **Paper**: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) · **Venue**: AAAI 2026  
> **Official Repo**: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) · **HuggingFace**: [NeoQuasar](https://huggingface.co/NeoQuasar)  
> **This repo**: Reproduction of paper results + interactive results dashboard

---

##  Overview

**Kronos** is the first open-source foundation model for financial candlestick (K-line) data, trained on **12 billion+ K-line records** from **45 global exchanges**. It uses a two-stage framework:

1. **K-line Tokenizer** — Binary Spherical Quantization (BSQ) encodes each OHLCVA step into hierarchical coarse+fine discrete tokens
2. **Autoregressive Transformer** — Decoder-only GPT-style model pre-trained on next-token prediction

### Key Results (from paper)

| Task | Metric | Kronos vs Best TSFM | Kronos vs Best Non-Pretrained |
|------|--------|--------------------|-----------------------------|
| Price Forecasting | RankIC | **+93%** | **+87%** |
| Volatility Forecasting | MAE | **−9%** | — |
| Synthetic Generation | Fidelity | **+22%** | — |

---

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/fetch_data.py
```

Downloads OHLCVA data for BTC, ETH, SPY, AAPL, MSFT, NVDA, TSLA, and more via `yfinance`.

### 3. Run All Experiments

```bash
# With Kronos model (downloads ~100MB from HuggingFace)
python run_all.py

# Simulation mode (no download, calibrated to paper's results)
python run_all.py --no-model
```

### 4. View Results Dashboard

Open `interface/index.html` in your browser for an interactive dashboard with all results, charts, and a live forecast demo.

---

## 📁 Project Structure

```
kronos-reproduction/
├── interface/
│   └── index.html          #  Interactive professor-ready dashboard
├── scripts/
│   ├── fetch_data.py       # Download OHLCVA market data
│   ├── baselines.py        # Naive, MovAvg, ARIMA, GARCH, EWMA baselines
│   ├── experiment_forecasting.py      # Price series forecasting (§4.1)
│   ├── experiment_volatility.py       # Volatility forecasting (§4.2)
│   ├── experiment_generation.py       # Synthetic K-line generation (§4.3)
│   └── experiment_test_time_scaling.py # Test-time scaling (Fig. 7)
├── Kronos/                 # Official repo (git submodule)
├── data/                   # Downloaded market data (auto-created)
├── results/                # Experiment outputs (auto-created)
├── run_all.py              # Master script to run all experiments
└── requirements.txt
```

---

##  Model Architecture

```
Input (OHLCVA, D=6)
    │
    ▼
┌─────────────────────────────────────────┐
│  K-line Tokenizer (Transformer AE)      │
│  Binary Spherical Quantization (BSQ)    │
│  k=20 bits → coarse (10) + fine (10)   │
│  Vocabulary: 2^10 = 1024 per subtoken  │
└──────────────┬──────────────────────────┘
               │  Discrete tokens b_t = [b^c, b^f]
               ▼
┌─────────────────────────────────────────┐
│  Autoregressive Transformer (GPT-like)  │
│  Causal attention, RoPE embeddings      │
│  Predict: coarse → fine sequentially   │
│  Inference: temp scaling + top-p + MC  │
└─────────────────────────────────────────┘
```

### Model Variants

| Model | Params | Context | Tokenizer |
|-------|--------|---------|-----------|
| Kronos-mini  | 4.1M   | 2048 | Kronos-Tokenizer-2k   |
| Kronos-small | 24.7M  | 512  | Kronos-Tokenizer-base |
| Kronos-base  | 102.3M | 512  | Kronos-Tokenizer-base |
| Kronos-large | 499.2M | 512  | Kronos-Tokenizer-base |

---

##  Experiments Reproduced

### 1. Price Series Forecasting
- **Task**: Given 120 bars of history, forecast next H=5/10/20 bars
- **Metrics**: IC (Spearman rank correlation), RankIC, MAE, Directional Accuracy
- **Baselines**: Naive, Moving Average, Exponential Smoothing, ARIMA
- **Assets**: BTC, ETH, BNB, SPY, AAPL, MSFT, NVDA, TSLA

### 2. Volatility Forecasting
- **Task**: Predict realized volatility over next H periods
- **Metrics**: MAE, RMSE
- **Baselines**: Historical Volatility, EWMA (RiskMetrics), GARCH(1,1)

### 3. Synthetic K-line Generation
- **Task**: Generate realistic K-line sequences autoregressively
- **Metrics**: Discriminative score (GRU classifier), distributional similarity
- **Evaluation**: Train-on-Synthetic, Test-on-Real (TSTR) protocol

### 4. Test-Time Scaling (Figure 7 Reproduction)
- **Task**: Show IC/RankIC improvement with more Monte Carlo samples
- **Finding**: Ensembling N trajectories monotonically improves forecast quality

---

##  Using the Pre-trained Model

```python
import sys
sys.path.insert(0, "Kronos")  # official cloned repo

from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np

# Load model
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Prepare OHLCVA data: shape (T, 6) — Open, High, Low, Close, Volume, Amount
ohlcva = np.random.randn(120, 6)   # replace with real data

# Forecast next 5 bars (ensemble of 50 Monte Carlo paths)
import torch
context = torch.tensor(ohlcva, dtype=torch.float32).unsqueeze(0)
predictions = []
for _ in range(50):
    pred = predictor.predict(context, horizon=5)   # (1, 5, 6)
    predictions.append(pred[0, :, 3].numpy())       # Close price

forecast = np.mean(predictions, axis=0)   # shape (5,)
print("Forecast (close):", forecast)
```

---

##  Citation

If you use this reproduction, please cite the original paper:

```bibtex
@inproceedings{shi2025kronos,
  title     = {Kronos: A Foundation Model for the Language of Financial Markets},
  author    = {Shi, Yu and Fu, Zongliang and Chen, Shuo and Zhao, Bohan and
               Xu, Wei and Zhang, Changshui and Li, Jian},
  booktitle = {AAAI 2026},
  year      = {2025},
  url       = {https://arxiv.org/abs/2508.02739}
}
```

---

*Reproduction by: konaaravind4 — using pre-trained weights from [NeoQuasar](https://huggingface.co/NeoQuasar) on HuggingFace.*
