"""
metrics.py — Centralised evaluation metrics for Kronos reproduction experiments.

Provides:
    - Standalone metric functions (IC, RankIC, MAE, MSE, DA, Sharpe, Sortino,
      Calmar, MaxDrawdown, VaR).
    - ``MetricsTracker`` — accumulates per-window results across experiments and
      can export to KonaDB or CSV.
    - ``compare_models`` — builds a tidy comparison DataFrame from a nested
      results dictionary.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Re-export canonical metric functions from baselines ─────────────────────

from scripts.baselines import (
    information_coefficient as ic,
    rank_ic,
    directional_accuracy,
    mae,
    mse,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    value_at_risk,
)

__all__ = [
    "ic",
    "rank_ic",
    "directional_accuracy",
    "mae",
    "mse",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "value_at_risk",
    "compute_all_metrics",
    "MetricsTracker",
    "compare_models",
]


# ─── Composite helper ────────────────────────────────────────────────────────

def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    risk_free: float = 0.0,
    var_confidence: float = 0.95,
) -> dict[str, float]:
    """Compute a comprehensive set of evaluation metrics.

    Given arrays of *return* values (not raw prices), compute:
    - IC (Spearman), RankIC, MAE, MSE, Directional Accuracy
    - A simple long-only Sharpe Ratio on implied trading strategy
    - VaR at the given confidence level

    Args:
        actual:          1-D array of realised returns.
        predicted:       1-D array of predicted returns.
        risk_free:       Daily risk-free rate used for Sharpe calculation.
        var_confidence:  Confidence level for VaR (default 0.95).

    Returns:
        Dictionary mapping metric names to float values.
    """
    n = min(len(actual), len(predicted))
    if n < 2:
        logger.warning("compute_all_metrics: fewer than 2 observations; "
                       "returning zeros.")
        return {k: 0.0 for k in
                ["IC", "RankIC", "MAE", "MSE", "DirectionalAccuracy",
                 "SharpeRatio", "VaR"]}

    a = actual[:n]
    p = predicted[:n]

    # Implied strategy: go long when prediction is positive, flat otherwise
    strategy_returns = np.where(p > 0, a, 0.0)

    metrics: dict[str, float] = {
        "IC":                 ic(a, p),
        "RankIC":             rank_ic(a, p),
        "MAE":                mae(a, p),
        "MSE":                mse(a, p),
        "DirectionalAccuracy": directional_accuracy(a, p),
        "SharpeRatio":        sharpe_ratio(strategy_returns, risk_free),
        "VaR":                value_at_risk(a, var_confidence),
    }

    logger.debug("compute_all_metrics result: %s", metrics)
    return metrics


# ─── MetricsTracker ──────────────────────────────────────────────────────────

@dataclass
class MetricsTracker:
    """Accumulate per-window metric values across rolling experiments.

    Usage::

        tracker = MetricsTracker(experiment="my_exp", model="Kronos", asset="BTC_USD")
        for window in windows:
            m = compute_all_metrics(actual_window, pred_window)
            tracker.update(m)
        summary = tracker.summary()
        tracker.to_csv("results/metrics.csv")
        tracker.to_kona_db("results/experiments.db")

    Attributes:
        experiment: Human-readable experiment tag (e.g. "forecasting_H5").
        model:      Model name (e.g. "Kronos", "Naive").
        asset:      Asset ticker (e.g. "BTC_USD").
        horizon:    Forecast horizon.
        _records:   Internal list of per-window metric dicts.
    """

    experiment: str
    model: str
    asset: str
    horizon: int = 0
    _records: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def update(self, metrics: dict[str, float]) -> None:
        """Append a new window's metric dict to the internal record list.

        Args:
            metrics: Dictionary of metric name → float, as returned by
                     :func:`compute_all_metrics`.
        """
        self._records.append(metrics)

    def summary(self) -> dict[str, float]:
        """Aggregate all accumulated records by taking the mean per metric.

        Returns:
            Dictionary of {metric_name: mean_value}.  Returns an empty dict if
            no records have been added.
        """
        if not self._records:
            return {}
        df = pd.DataFrame(self._records)
        result: dict[str, float] = df.mean(numeric_only=True).to_dict()
        result["n_windows"] = float(len(self._records))
        return result

    def to_csv(self, path: str, append: bool = True) -> None:
        """Export aggregated summary row to a CSV file.

        Args:
            path:   Destination CSV file path.
            append: If True and file exists, append a new row without header.
                    Otherwise, write a fresh file.
        """
        summary = self.summary()
        if not summary:
            logger.warning("MetricsTracker.to_csv: no data to export.")
            return

        row: dict[str, Any] = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": self.experiment,
            "model":      self.model,
            "asset":      self.asset,
            "horizon":    self.horizon,
            **summary,
        }

        import os
        write_header = not (append and os.path.exists(path))
        with open(path, "a" if append else "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info("MetricsTracker: exported summary to %s", path)

    def to_kona_db(self, db_path: str) -> None:
        """Persist the aggregated summary into a KonaDB table.

        Creates the ``experiment_metrics`` table if it does not exist, then
        inserts the current summary as a new row.

        Args:
            db_path: File path to the KonaDB database.
        """
        try:
            import kona  # type: ignore[import]
        except ImportError:
            logger.error(
                "MetricsTracker.to_kona_db: 'kona' package not installed.  "
                "Install it with: pip install kona-db"
            )
            return

        summary = self.summary()
        if not summary:
            logger.warning("MetricsTracker.to_kona_db: no data to export.")
            return

        row: dict[str, Any] = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "experiment": self.experiment,
            "model":      self.model,
            "asset":      self.asset,
            "horizon":    self.horizon,
            **summary,
        }

        db = kona.DB(db_path)
        db.create_table_if_not_exists(
            "experiment_metrics",
            schema={
                "id":          "INTEGER PRIMARY KEY AUTOINCREMENT",
                "timestamp":   "TEXT",
                "experiment":  "TEXT",
                "model":       "TEXT",
                "asset":       "TEXT",
                "horizon":     "INTEGER",
                "IC":          "REAL",
                "RankIC":      "REAL",
                "MAE":         "REAL",
                "MSE":         "REAL",
                "DirectionalAccuracy": "REAL",
                "SharpeRatio": "REAL",
                "VaR":         "REAL",
                "n_windows":   "INTEGER",
            },
        )
        db.insert("experiment_metrics", row)
        logger.info(
            "MetricsTracker: persisted summary for %s/%s to KonaDB at %s",
            self.model, self.asset, db_path,
        )


# ─── Model comparison table ───────────────────────────────────────────────────

def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a tidy comparison DataFrame from a nested results dictionary.

    Args:
        results: Nested dict with structure::

                     {
                         "ModelA": {"IC": 0.12, "RankIC": 0.11, ...},
                         "ModelB": {"IC": 0.08, "RankIC": 0.07, ...},
                     }

    Returns:
        A :class:`pandas.DataFrame` with one row per model, sorted
        descending by ``RankIC`` (if present).

    Example::

        df = compare_models({
            "Kronos":  tracker_kronos.summary(),
            "Naive":   tracker_naive.summary(),
            "ARIMA":   tracker_arima.summary(),
        })
        print(df.to_string())
    """
    if not results:
        logger.warning("compare_models: empty results dict; returning empty DataFrame.")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        row: dict[str, Any] = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Sort by RankIC descending if available
    if "RankIC" in df.columns:
        df = df.sort_values("RankIC", ascending=False)

    # Round numeric columns for readability
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    logger.info("compare_models: built comparison table for %d models.", len(df))
    return df
