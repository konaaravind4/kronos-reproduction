"""
fetch_data.py — Download real OHLCVA market data using yfinance
Formats data to match Kronos's D=6 input: Open, High, Low, Close, Volume, Amount
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CRYPTO_ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
STOCK_ASSETS  = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN"]

ALL_ASSETS = CRYPTO_ASSETS + STOCK_ASSETS

def fetch_ohlcva(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data and compute Amount = Volume * Close (proxy for turnover).
    Returns a DataFrame with columns: [open, high, low, close, volume, amount]
    """
    print(f"  Downloading {ticker} ({interval}, {period})...")
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
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

        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:]
    )


def main():
    print("=" * 60)
    print("  Kronos Paper — Data Acquisition")
    print("  Downloading OHLCVA data for experiments")
    print("=" * 60)

    summary = []

    for ticker in ALL_ASSETS:
        df = fetch_ohlcva(ticker, period="5y", interval="1d")
        if df is None:
            continue

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
