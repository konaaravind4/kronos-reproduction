"""
fetch_data.py — Download real OHLCVA market data using yfinance.
Formats data to match Kronos's D=6 input: Open, High, Low, Close, Volume, Amount.

New utilities:
    - validate_ohlcva   : data-quality checks
    - compute_returns   : simple / log return series
    - resample_ohlcv    : resample to an arbitrary timeframe
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CRYPTO_ASSETS: list[str] = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
STOCK_ASSETS:  list[str] = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN"]

ALL_ASSETS: list[str] = CRYPTO_ASSETS + STOCK_ASSETS


# ─── Core download ───────────────────────────────────────────────────────────

def fetch_ohlcva(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Download OHLCV data and compute Amount = Volume × Close (proxy for turnover).

    Returns a DataFrame with lowercase columns:
    ``[open, high, low, close, volume, amount]``.

    Args:
        ticker:   Yahoo Finance ticker symbol (e.g. ``"BTC-USD"`` or ``"AAPL"``).
        period:   Download period string accepted by ``yfinance`` (e.g. ``"5y"``).
        interval: Bar interval (e.g. ``"1d"``, ``"1h"``, ``"15m"``).

    Returns:
        Cleaned DataFrame, or ``None`` if the download fails or returns no data.
    """
    logger.info("Downloading %s (interval=%s, period=%s)...", ticker, interval, period)
    print(f"  Downloading {ticker} ({interval}, {period})...")
    try:
        df: pd.DataFrame = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=True, progress=False,
        )
        if df.empty:
            logger.warning("No data returned for %s", ticker)
            print(f"  [WARN] No data for {ticker}")
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df["amount"] = df["volume"] * df["close"]   # Amount / Turnover

        # Drop rows with any NaN
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"

        # Ensure positive values
        df = df[(df["close"] > 0) & (df["volume"] > 0)]

        logger.info(
            "%s: %d bars, %s → %s",
            ticker, len(df), df.index[0].date(), df.index[-1].date(),
        )
        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as exc:
        logger.error("Error downloading %s: %s", ticker, exc)
        print(f"  [ERROR] {ticker}: {exc}")
        return None


# ─── Data validation ─────────────────────────────────────────────────────────

def validate_ohlcva(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate an OHLCVA DataFrame for common data-quality issues.

    Checks performed:
    1. Required columns are present.
    2. No missing (NaN) values in any column.
    3. No zero-volume bars.
    4. Close prices are strictly positive.
    5. Price anomalies: daily move > 50% (likely data error).
    6. High ≥ Low on all bars.
    7. High ≥ Close and Low ≤ Close on all bars.

    Args:
        df: DataFrame with lowercase OHLCVA columns
            ``[open, high, low, close, volume, amount]``.

    Returns:
        Tuple ``(is_valid, issues)`` where ``is_valid`` is ``True`` only if no
        issues were found, and ``issues`` is a list of human-readable
        problem descriptions.
    """
    required_cols = {"open", "high", "low", "close", "volume", "amount"}
    issues: list[str] = []

    # 1. Column presence
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {sorted(missing_cols)}")
        # Can't run further checks without required columns
        return False, issues

    # 2. NaN check
    nan_counts = df[list(required_cols)].isna().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            issues.append(f"Column '{col}' has {cnt} NaN value(s).")

    # 3. Zero-volume bars
    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol > 0:
        issues.append(f"{zero_vol} bar(s) have zero volume.")

    # 4. Non-positive close prices
    non_pos_close = int((df["close"] <= 0).sum())
    if non_pos_close > 0:
        issues.append(f"{non_pos_close} bar(s) have non-positive close price.")

    # 5. Price anomalies (>50% single-bar move)
    if len(df) > 1:
        pct_change = df["close"].pct_change().abs()
        extreme_moves = int((pct_change > 0.5).sum())
        if extreme_moves > 0:
            issues.append(
                f"{extreme_moves} bar(s) show >50% single-bar price change "
                "(possible data anomaly)."
            )

    # 6. High ≥ Low
    hl_violation = int((df["high"] < df["low"]).sum())
    if hl_violation > 0:
        issues.append(f"{hl_violation} bar(s) have high < low.")

    # 7. High ≥ Close and Low ≤ Close
    hc_violation = int((df["high"] < df["close"]).sum())
    lc_violation = int((df["low"] > df["close"]).sum())
    if hc_violation > 0:
        issues.append(f"{hc_violation} bar(s) have high < close.")
    if lc_violation > 0:
        issues.append(f"{lc_violation} bar(s) have low > close.")

    is_valid = len(issues) == 0
    if is_valid:
        logger.info("validate_ohlcva: data passed all checks (%d bars).", len(df))
    else:
        logger.warning(
            "validate_ohlcva: %d issue(s) found:\n  %s",
            len(issues), "\n  ".join(issues),
        )
    return is_valid, issues


# ─── Return computation ───────────────────────────────────────────────────────

def compute_returns(
    df: pd.DataFrame,
    col: str = "close",
    method: str = "simple",
) -> pd.Series:
    """Compute a return series from a price column.

    Args:
        df:     DataFrame containing the price column.
        col:    Column name to compute returns from (default ``"close"``).
        method: ``"simple"`` for arithmetic returns ``(P_t / P_{t-1}) - 1``,
                or ``"log"`` for log-returns ``log(P_t / P_{t-1})``.

    Returns:
        :class:`pandas.Series` of returns with the same index as ``df``,
        with the first value as NaN (due to differencing).

    Raises:
        ValueError: If ``col`` is not in ``df.columns`` or ``method`` is
                    not ``"simple"`` or ``"log"``.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if method not in ("simple", "log"):
        raise ValueError(f"method must be 'simple' or 'log', got '{method}'.")

    prices = df[col].astype(float)
    if method == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices / prices.shift(1))

    logger.debug(
        "compute_returns: method=%s, col=%s, n=%d", method, col, len(rets)
    )
    return rets


# ─── Resampling ──────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Resample an OHLCV(A) DataFrame to a coarser timeframe.

    The ``amount`` column (if present) is summed over each new bar, matching
    turnover aggregation conventions.

    Args:
        df:   Source DataFrame with a ``datetime``-like index and at minimum
              the columns ``[open, high, low, close, volume]``.
        freq: Pandas offset alias for the target frequency, e.g. ``"1h"``,
              ``"4h"``, ``"1W"``.

    Returns:
        Resampled DataFrame with NaN-free rows (bars with all-NaN are dropped).

    Raises:
        ValueError: If required OHLCV columns are missing.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"resample_ohlcv: missing columns: {sorted(missing)}")

    agg: dict[str, str] = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    if "amount" in df.columns:
        agg["amount"] = "sum"

    resampled = df.resample(freq).agg(agg).dropna(how="all")
    logger.info(
        "resample_ohlcv: %d bars → %d bars (freq=%s)",
        len(df), len(resampled), freq,
    )
    return resampled


# ─── Train / val / test split ────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train / validation / test subsets chronologically.

    Args:
        df:          Source DataFrame (rows sorted chronologically).
        train_ratio: Fraction of rows allocated to training.
        val_ratio:   Fraction of rows allocated to validation.
        test_ratio:  Fraction of rows allocated to testing.

    Returns:
        Tuple ``(train_df, val_df, test_df)``.
    """
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:],
    )


# ─── Main download pipeline ──────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """Download all configured assets, validate, split, and save to disk.

    Returns:
        Summary :class:`pandas.DataFrame` with one row per asset.
    """
    print("=" * 60)
    print("  Kronos Paper — Data Acquisition")
    print("  Downloading OHLCVA data for experiments")
    print("=" * 60)

    summary: list[dict] = []

    for ticker in ALL_ASSETS:
        df = fetch_ohlcva(ticker, period="5y", interval="1d")
        if df is None:
            continue

        # Data quality check
        is_valid, issues = validate_ohlcva(df)
        if not is_valid:
            print(f"  [WARN] {ticker} data quality issues:")
            for issue in issues:
                print(f"         - {issue}")

        # Save full dataset
        out_path = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}_daily.csv")
        df.to_csv(out_path)

        # Train/val/test split
        train, val, test = split_data(df)
        train.to_csv(os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}_train.csv"))
        val.to_csv(os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}_val.csv"))
        test.to_csv(os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}_test.csv"))

        summary.append({
            "ticker":     ticker,
            "total_bars": len(df),
            "train_bars": len(train),
            "val_bars":   len(val),
            "test_bars":  len(test),
            "start":      str(df.index[0].date()),
            "end":        str(df.index[-1].date()),
            "avg_close":  round(df["close"].mean(), 2),
            "data_valid": is_valid,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(DATA_DIR, "dataset_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n✅ Data saved to: {DATA_DIR}")
    return summary_df


if __name__ == "__main__":
    main()
