"""
baselines.py — Lightweight baseline implementations for comparison with Kronos
Includes: Naive, Moving Average, ARIMA, GARCH, Historical Volatility,
          Linear Regression Trend, Ensemble, and Financial Risk Metrics.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ─── Price Forecasting Baselines ─────────────────────────────────────────────

def naive_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Last-value (random walk) forecast.

    Args:
        series:  1-D array of historical prices.
        horizon: Number of future steps to forecast.

    Returns:
        Array of shape (horizon,) filled with the last observed value.
    """
    return np.full(horizon, series[-1])


def moving_average_forecast(series: np.ndarray, horizon: int,
                             window: int = 20) -> np.ndarray:
    """Simple moving average forecast.

    Args:
        series:  1-D array of historical prices.
        horizon: Number of future steps to forecast.
        window:  Look-back window for the mean calculation.

    Returns:
        Array of shape (horizon,) filled with the rolling mean.
    """
    ma = np.mean(series[-window:])
    return np.full(horizon, ma)


def exponential_smoothing_forecast(series: np.ndarray, horizon: int,
                                    alpha: float = 0.3) -> np.ndarray:
    """Exponential smoothing (ETS) forecast.

    Args:
        series:  1-D array of historical prices.
        horizon: Number of future steps to forecast.
        alpha:   Smoothing coefficient in (0, 1).

    Returns:
        Array of shape (horizon,) filled with the smoothed level.
    """
    smoothed: float = float(series[0])
    for val in series[1:]:
        smoothed = alpha * float(val) + (1 - alpha) * smoothed
    return np.full(horizon, smoothed)


def arima_forecast(series: np.ndarray, horizon: int) -> Optional[np.ndarray]:
    """ARIMA(1,1,1) forecast for close price.

    Falls back to naive forecast if ``statsmodels`` is unavailable or fitting
    fails.

    Args:
        series:  1-D array of historical prices.
        horizon: Number of future steps to forecast.

    Returns:
        Array of shape (horizon,) with point forecasts.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        fit   = model.fit()
        pred  = fit.forecast(steps=horizon)
        return np.array(pred)
    except Exception as exc:
        logger.warning("ARIMA fitting failed (%s); falling back to naive.", exc)
        return naive_forecast(series, horizon)


def garch_volatility_forecast(returns: np.ndarray, horizon: int) -> np.ndarray:
    """GARCH(1,1) approximation for volatility forecasting.

    Attempts to fit a proper GARCH(1,1) model via the ``arch`` library.
    If that fails, falls back to an analytical GARCH(1,1) recursion with
    MoM-estimated parameters (omega, alpha, beta).

    Args:
        returns: 1-D array of log-returns (or simple returns).
        horizon: Number of future steps to forecast.

    Returns:
        Array of shape (horizon,) with annualised volatility forecasts.
    """
    try:
        from arch import arch_model
        am  = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off")
        fc  = res.forecast(horizon=horizon, reindex=False)
        vols = np.sqrt(fc.variance.values[-1]) / 100 * np.sqrt(252)
        logger.debug("GARCH(1,1) fit succeeded; horizon=%d", horizon)
        return vols
    except Exception as exc:
        logger.warning("arch GARCH fit failed (%s); using analytical GARCH.", exc)

    # ── Analytical GARCH(1,1) approximation ──────────────────────────────────
    r2 = returns ** 2
    var_long = float(np.var(returns, ddof=1))
    alpha_hat = 0.10
    beta_hat  = 0.85
    omega_hat = var_long * (1 - alpha_hat - beta_hat)
    omega_hat = max(omega_hat, 1e-10)

    h_t = var_long
    forecasts: list[float] = []
    for step in range(horizon):
        if step == 0:
            h_t = omega_hat + alpha_hat * float(r2[-1]) + beta_hat * h_t
        else:
            h_t = omega_hat + (alpha_hat + beta_hat) * h_t
        forecasts.append(np.sqrt(max(h_t, 0.0)) * np.sqrt(252))
    return np.array(forecasts)


def linear_regression_forecast(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Simple OLS (ordinary least squares) linear trend forecast.

    Fits a straight line through the historical price series and extrapolates
    it by ``horizon`` steps.

    Args:
        prices:  1-D array of historical prices.
        horizon: Number of future steps to forecast.

    Returns:
        Array of shape (horizon,) with extrapolated trend values.
    """
    n = len(prices)
    x = np.arange(n, dtype=float)
    # OLS coefficients
    x_mean = x.mean()
    y_mean = prices.mean()
    slope = float(np.sum((x - x_mean) * (prices - y_mean)) /
                  (np.sum((x - x_mean) ** 2) + 1e-12))
    intercept = y_mean - slope * x_mean

    future_x = np.arange(n, n + horizon, dtype=float)
    forecast  = intercept + slope * future_x
    logger.debug("OLS forecast: slope=%.6f, intercept=%.4f", slope, intercept)
    return forecast


def ensemble_forecast(*forecasts: np.ndarray,
                       weights: Optional[list[float]] = None) -> np.ndarray:
    """Weighted average of multiple individual forecasts.

    Args:
        *forecasts: Variable number of 1-D forecast arrays, all with the same
                    length.
        weights:    Optional list of floats with the same length as the number
                    of forecasts.  If ``None``, equal weights are used.
                    Weights are normalised to sum to 1.

    Returns:
        Array of shape ``(horizon,)`` containing the blended forecast.

    Raises:
        ValueError: If ``weights`` length does not match the number of
                    forecasts, or if all weights are zero.
    """
    if not forecasts:
        raise ValueError("At least one forecast must be provided.")

    n_models = len(forecasts)
    if weights is None:
        w = np.ones(n_models) / n_models
    else:
        if len(weights) != n_models:
            raise ValueError(
                f"len(weights)={len(weights)} != n_forecasts={n_models}"
            )
        w_arr = np.array(weights, dtype=float)
        total = w_arr.sum()
        if total == 0:
            raise ValueError("Sum of weights must not be zero.")
        w = w_arr / total

    stacked = np.stack(forecasts, axis=0)           # (n_models, horizon)
    blended = np.einsum("m,mh->h", w, stacked)      # weighted sum
    logger.debug("Ensemble of %d models with weights %s", n_models, w.tolist())
    return blended


# ─── Volatility Baselines ─────────────────────────────────────────────────────

def historical_volatility(log_returns: np.ndarray, horizon: int,
                           window: int = 20) -> np.ndarray:
    """Historical volatility (annualized std of log-returns).

    Args:
        log_returns: 1-D array of log-returns.
        horizon:     Number of future steps to forecast.
        window:      Look-back window.

    Returns:
        Array of shape (horizon,) filled with the annualised historical vol.
    """
    vol = np.std(log_returns[-window:]) * np.sqrt(252)
    return np.full(horizon, vol)


def ewma_volatility(log_returns: np.ndarray, horizon: int,
                    lam: float = 0.94) -> np.ndarray:
    """EWMA (RiskMetrics) volatility forecast.

    Args:
        log_returns: 1-D array of log-returns.
        horizon:     Number of future steps to forecast.
        lam:         Decay factor (λ) for EWMA; typically 0.94 for daily data.

    Returns:
        Array of shape (horizon,) filled with the annualised EWMA vol.
    """
    var: float = float(log_returns[0] ** 2)
    for r in log_returns[1:]:
        var = lam * var + (1 - lam) * float(r) ** 2
    vol = np.sqrt(var * 252)
    return np.full(horizon, vol)


def garch_volatility(log_returns: np.ndarray,
                     horizon: int) -> Optional[np.ndarray]:
    """GARCH(1,1) volatility forecast.

    Args:
        log_returns: 1-D array of log-returns.
        horizon:     Number of future steps to forecast.

    Returns:
        Array of shape (horizon,) with GARCH volatility forecasts.
        Falls back to EWMA if the ``arch`` package is unavailable.
    """
    try:
        from arch import arch_model
        model = arch_model(log_returns * 100, vol="Garch", p=1, q=1,
                           rescale=False)
        fit   = model.fit(disp="off")
        fcast = fit.forecast(horizon=horizon)
        vols  = np.sqrt(fcast.variance.values[-1]) / 100 * np.sqrt(252)
        return vols
    except Exception as exc:
        logger.warning("garch_volatility fallback to EWMA: %s", exc)
        return ewma_volatility(log_returns, horizon)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (IC) between predictions and true returns.

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.

    Returns:
        IC value in [-1, 1].  Returns 0.0 for degenerate inputs.
    """
    from scipy.stats import spearmanr
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RankIC: Pearson correlation of ranks.

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.

    Returns:
        RankIC value in [-1, 1].  Returns 0.0 for degenerate inputs.
    """
    from scipy.stats import rankdata
    if len(y_true) < 2:
        return 0.0
    r_true = rankdata(y_true)
    r_pred = rankdata(y_pred)
    corr   = np.corrcoef(r_true, r_pred)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with the correct sign.

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.

    Returns:
        Directional accuracy in [0, 1].
    """
    if len(y_true) < 2:
        return 0.5
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    return float(np.mean(signs_true == signs_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.

    Returns:
        MSE as a float.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Args:
        y_true: 1-D array of realised values.
        y_pred: 1-D array of predicted values.
        eps:    Small constant to avoid division-by-zero.

    Returns:
        MAPE value as a percentage float.
    """
    return float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100
    )


# ─── Financial Risk & Performance Metrics ────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sharpe Ratio.

    Computed as (mean_excess_return / std_return) * sqrt(252), where 252 is the
    standard number of trading days per year.

    Args:
        returns:    1-D array of daily returns (simple or log).
        risk_free:  Daily risk-free rate (default 0.0).

    Returns:
        Annualised Sharpe Ratio.  Returns 0.0 for zero-variance returns.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std    = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    sr = float(np.mean(excess)) / std * np.sqrt(252)
    logger.debug("Sharpe ratio: %.4f", sr)
    return sr


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised Sortino Ratio.

    Uses the downside deviation (std of negative excess returns only) as the
    denominator, rewarding upside volatility less harshly.

    Args:
        returns:    1-D array of daily returns (simple or log).
        risk_free:  Daily risk-free rate (default 0.0).

    Returns:
        Annualised Sortino Ratio.  Returns 0.0 when downside deviation is zero.
    """
    if len(returns) < 2:
        return 0.0
    excess      = returns - risk_free
    downside    = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std    = float(np.std(downside, ddof=1))
    if down_std == 0.0:
        return 0.0
    sr = float(np.mean(excess)) / down_std * np.sqrt(252)
    logger.debug("Sortino ratio: %.4f", sr)
    return sr


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a price series.

    Args:
        prices: 1-D array of price levels (must be positive).

    Returns:
        Maximum drawdown as a positive fraction in [0, 1].
        Returns 0.0 for a monotonically increasing series.
    """
    if len(prices) < 2:
        return 0.0
    peak    = np.maximum.accumulate(prices)
    dd      = (prices - peak) / (peak + 1e-12)
    mdd     = float(-np.min(dd))
    logger.debug("Max drawdown: %.4f", mdd)
    return mdd


def calmar_ratio(returns: np.ndarray, max_drawdown_val: float) -> float:
    """Calmar Ratio: annualised return divided by maximum drawdown.

    Args:
        returns:          1-D array of daily returns.
        max_drawdown_val: Maximum drawdown as a positive fraction (e.g. 0.25
                          for 25%).  Use :func:`max_drawdown` to compute it.

    Returns:
        Calmar Ratio.  Returns 0.0 when ``max_drawdown_val`` is zero.
    """
    if max_drawdown_val <= 0:
        return 0.0
    ann_return = float(np.mean(returns)) * 252
    cr = ann_return / max_drawdown_val
    logger.debug("Calmar ratio: %.4f", cr)
    return cr


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical (non-parametric) Value-at-Risk at the given confidence level.

    Returns the loss threshold such that losses exceed this value with
    probability ``1 - confidence``.

    Args:
        returns:    1-D array of daily returns (simple or log).
        confidence: Confidence level, e.g. 0.95 for 95% VaR.

    Returns:
        VaR as a **positive** number representing the potential loss.
        E.g. 0.02 means "95% of the time, daily loss is ≤ 2%".
    """
    if len(returns) < 2:
        return 0.0
    var = float(-np.percentile(returns, (1 - confidence) * 100))
    logger.debug("VaR(%.0f%%): %.6f", confidence * 100, var)
    return var
