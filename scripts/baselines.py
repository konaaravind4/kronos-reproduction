"""
baselines.py — Lightweight baseline implementations for comparison with Kronos
Includes: Naive, Moving Average, ARIMA, GARCH, Historical Volatility
"""
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Price Forecasting Baselines ─────────────────────────────────────────────

def naive_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Last-value (random walk) forecast."""
    return np.full(horizon, series[-1])


def moving_average_forecast(series: np.ndarray, horizon: int, window: int = 20) -> np.ndarray:
    """Simple moving average forecast."""
    ma = np.mean(series[-window:])
    return np.full(horizon, ma)


def exponential_smoothing_forecast(series: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Exponential smoothing (ETS) forecast."""
    smoothed = series[0]
    for val in series[1:]:
        smoothed = alpha * val + (1 - alpha) * smoothed
    return np.full(horizon, smoothed)


def arima_forecast(series: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    """ARIMA(1,1,1) forecast for close price."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        fit   = model.fit()
        pred  = fit.forecast(steps=horizon)
        return np.array(pred)
    except Exception:
        return naive_forecast(series, horizon)


# ─── Volatility Baselines ─────────────────────────────────────────────────────

def historical_volatility(log_returns: np.ndarray, horizon: int, window: int = 20) -> np.ndarray:
    """Historical volatility (annualized std of log-returns)."""
    vol = np.std(log_returns[-window:]) * np.sqrt(252)
    return np.full(horizon, vol)


def ewma_volatility(log_returns: np.ndarray, horizon: int, lam: float = 0.94) -> np.ndarray:
    """EWMA (RiskMetrics) volatility forecast."""
    var = log_returns[0] ** 2
    for r in log_returns[1:]:
        var = lam * var + (1 - lam) * r ** 2
    vol = np.sqrt(var * 252)
    return np.full(horizon, vol)


def garch_volatility(log_returns: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    """GARCH(1,1) volatility forecast."""
    try:
        from arch import arch_model
        model = arch_model(log_returns * 100, vol="Garch", p=1, q=1, rescale=False)
        fit   = model.fit(disp="off")
        fcast = fit.forecast(horizon=horizon)
        vols  = np.sqrt(fcast.variance.values[-1]) / 100 * np.sqrt(252)
        return vols
    except Exception:
        return ewma_volatility(log_returns, horizon)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (IC) between predictions and true returns."""
    from scipy.stats import spearmanr
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RankIC: Pearson correlation of ranks."""
    from scipy.stats import rankdata
    if len(y_true) < 2:
        return 0.0
    r_true = rankdata(y_true)
    r_pred = rankdata(y_pred)
    corr   = np.corrcoef(r_true, r_pred)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign."""
    if len(y_true) < 2:
        return 0.5
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    return float(np.mean(signs_true == signs_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
