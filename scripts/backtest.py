"""
Kronos Backtesting Module
=========================
Portfolio backtesting simulation using Kronos-style forecasts.
Tests forecast-driven trading strategies against historical OHLCVA data.

Strategies supported:
- threshold_long  : go long when forecast > threshold
- threshold_short : go short when forecast < -threshold
- momentum        : follow forecast direction with confidence weighting
- sentiment_blend : blend Kronos forecast + sentiment signal from Dashboard

Results are stored in KonaDB for persistence and cross-experiment comparison.

Usage:
    from scripts.backtest import Backtester, ThresholdStrategy

    bt = Backtester(initial_capital=10000.0, kona_db="results/backtest.kona")
    results = bt.run(
        prices=[42000, 43100, 41800, 44200, 43900],
        forecasts=[43000, 41500, 44000, 43500, 45000],
        strategy=ThresholdStrategy(threshold=0.01)
    )
    print(results.summary())
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol


# ── Interfaces ────────────────────────────────────────────────────────────────

class Strategy(Protocol):
    """Trading strategy interface — takes a forecast and returns a position."""
    def decide(self, price: float, forecast: float, sentiment: float | None) -> float:
        """
        Return position size: +1.0 = full long, -1.0 = full short, 0.0 = flat.

        Args:
            price: Current market price.
            forecast: Model forecast for next period.
            sentiment: Optional sentiment signal (-1.0 to +1.0).
        """
        ...


# ── Strategies ────────────────────────────────────────────────────────────────

@dataclass
class ThresholdStrategy:
    """
    Go long if forecast > price * (1 + threshold), short if below -threshold.
    Simple directional strategy — baseline for comparison.
    """
    threshold: float = 0.005  # 0.5% minimum expected move

    def decide(self, price: float, forecast: float, sentiment: float | None = None) -> float:
        expected_return = (forecast - price) / price if price > 0 else 0.0
        if expected_return > self.threshold:
            return 1.0   # long
        elif expected_return < -self.threshold:
            return -1.0  # short
        return 0.0       # flat


@dataclass
class MomentumStrategy:
    """
    Position size proportional to forecast confidence.
    Stronger signals → larger positions.
    """
    max_position: float = 1.0
    confidence_scale: float = 2.0

    def decide(self, price: float, forecast: float, sentiment: float | None = None) -> float:
        if price <= 0:
            return 0.0
        ret = (forecast - price) / price
        position = math.tanh(ret * self.confidence_scale) * self.max_position
        return round(position, 4)


@dataclass
class SentimentBlendStrategy:
    """
    Blends Kronos forecast with market sentiment signal.
    Sentiment from Real-time-Sentiment-Intelligence-Dashboard.
    
    blend_weight: 0.0 = pure Kronos, 1.0 = pure sentiment
    """
    blend_weight: float = 0.3     # 30% sentiment, 70% Kronos
    threshold: float = 0.005

    def decide(self, price: float, forecast: float, sentiment: float | None = None) -> float:
        if price <= 0:
            return 0.0

        kronos_signal = (forecast - price) / price
        if sentiment is not None:
            # Blend: weighted average of Kronos return signal and sentiment
            blended = (1 - self.blend_weight) * kronos_signal + self.blend_weight * sentiment * abs(kronos_signal)
        else:
            blended = kronos_signal

        if blended > self.threshold:
            return min(blended / self.threshold, 1.0)
        elif blended < -self.threshold:
            return max(blended / self.threshold, -1.0)
        return 0.0


# ── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    step: int
    price: float
    forecast: float
    position: float       # -1.0 to +1.0
    pnl: float = 0.0
    cumulative_pnl: float = 0.0
    portfolio_value: float = 0.0
    sentiment: float | None = None


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    trades: list[Trade]
    initial_capital: float
    final_capital: float
    strategy_name: str
    asset: str = "UNKNOWN"
    start_time: float = field(default_factory=time.time)

    @property
    def total_return(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return self.total_return * 100

    @property
    def n_trades(self) -> int:
        return sum(1 for t in self.trades if t.position != 0.0)

    @property
    def win_rate(self) -> float:
        positive = sum(1 for t in self.trades if t.pnl > 0)
        total = sum(1 for t in self.trades if t.pnl != 0)
        return positive / total if total > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        peak = self.initial_capital
        max_dd = 0.0
        for trade in self.trades:
            val = trade.portfolio_value
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio (risk-free rate = 0)."""
        pnls = [t.pnl for t in self.trades if t.pnl != 0]
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        std  = math.sqrt(sum((p - mean) ** 2 for p in pnls) / len(pnls))
        return (mean / std) * math.sqrt(252) if std > 0 else 0.0

    def summary(self) -> str:
        """Human-readable backtest summary."""
        return (
            f"\n{'='*55}\n"
            f"  Backtest Results — {self.strategy_name}\n"
            f"  Asset: {self.asset}\n"
            f"{'='*55}\n"
            f"  Initial Capital : ${self.initial_capital:,.2f}\n"
            f"  Final Capital   : ${self.final_capital:,.2f}\n"
            f"  Total Return    : {self.total_return_pct:+.2f}%\n"
            f"  Total Trades    : {self.n_trades}\n"
            f"  Win Rate        : {self.win_rate:.1%}\n"
            f"  Max Drawdown    : {self.max_drawdown:.1%}\n"
            f"  Sharpe Ratio    : {self.sharpe_ratio:.3f}\n"
            f"{'='*55}"
        )

    def to_dict(self) -> dict:
        """Serialize for KonaDB storage."""
        return {
            "strategy": self.strategy_name,
            "asset": self.asset,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return_pct": round(self.total_return_pct, 4),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "run_timestamp": self.start_time,
        }


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    """
    Simulate trading performance using Kronos forecasts on historical data.

    Args:
        initial_capital: Starting portfolio value in USD.
        commission: Transaction cost as fraction of trade value (default: 0.1%).
        kona_db: Optional path to .kona file for persisting results.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,
        kona_db: str | None = None,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self._kona_db = kona_db

    def run(
        self,
        prices: list[float],
        forecasts: list[float],
        strategy: Strategy,
        sentiments: list[float] | None = None,
        asset: str = "ASSET",
    ) -> BacktestResults:
        """
        Run a backtest simulation.

        Args:
            prices:     Historical price series (length N).
            forecasts:  Model forecasts for each step (length N).
            strategy:   Trading strategy (ThresholdStrategy, MomentumStrategy, etc.).
            sentiments: Optional sentiment signal series aligned with prices (-1.0 to +1.0).
            asset:      Asset name for reporting (e.g. "BTC", "SPY").

        Returns:
            BacktestResults with all trade records and performance metrics.

        Example:
            >>> bt = Backtester(initial_capital=10000)
            >>> results = bt.run(prices, forecasts, ThresholdStrategy(threshold=0.01))
            >>> print(results.summary())
        """
        assert len(prices) == len(forecasts), "prices and forecasts must have equal length"
        if sentiments is not None:
            assert len(sentiments) == len(prices), "sentiments must align with prices"

        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        trades: list[Trade] = []
        cumulative_pnl = 0.0

        for step, (price, forecast) in enumerate(zip(prices, forecasts)):
            sentiment = sentiments[step] if sentiments else None
            new_position = strategy.decide(price, forecast, sentiment)

            # Close existing position if direction changed
            pnl = 0.0
            if position != 0.0 and position != new_position:
                pnl = position * (price - entry_price) * (capital / price)
                pnl -= abs(pnl) * self.commission
                capital += pnl
                cumulative_pnl += pnl
                position = 0.0

            # Open new position
            if new_position != 0.0 and position == 0.0:
                entry_price = price
                position = new_position
                # Commission on open
                capital -= abs(new_position) * price * self.commission

            trades.append(Trade(
                step=step,
                price=price,
                forecast=forecast,
                position=position,
                pnl=round(pnl, 4),
                cumulative_pnl=round(cumulative_pnl, 4),
                portfolio_value=round(capital, 2),
                sentiment=sentiment,
            ))

        # Close final position
        if position != 0.0 and len(prices) > 0:
            last_price = prices[-1]
            final_pnl = position * (last_price - entry_price) * (capital / last_price)
            capital += final_pnl
            cumulative_pnl += final_pnl

        results = BacktestResults(
            trades=trades,
            initial_capital=self.initial_capital,
            final_capital=round(capital, 2),
            strategy_name=type(strategy).__name__,
            asset=asset,
        )

        # Persist to KonaDB
        if self._kona_db:
            self._save_results(results)

        return results

    def compare_strategies(
        self,
        prices: list[float],
        forecasts: list[float],
        sentiments: list[float] | None = None,
        asset: str = "ASSET",
    ) -> dict[str, BacktestResults]:
        """
        Run all built-in strategies and return a comparison dict.

        Returns:
            Dict mapping strategy name → BacktestResults.
        """
        strategies = {
            "ThresholdStrategy": ThresholdStrategy(),
            "MomentumStrategy": MomentumStrategy(),
            "SentimentBlendStrategy": SentimentBlendStrategy(),
        }
        return {
            name: self.run(prices, forecasts, strat, sentiments, asset)
            for name, strat in strategies.items()
        }

    def _save_results(self, results: BacktestResults) -> None:
        """Persist backtest results to KonaDB."""
        try:
            import kona
            conn = kona.connect(self._kona_db)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    strategy VARCHAR(64),
                    asset VARCHAR(32),
                    initial_capital FLOAT,
                    final_capital FLOAT,
                    total_return_pct FLOAT,
                    n_trades INT,
                    win_rate FLOAT,
                    max_drawdown FLOAT,
                    sharpe_ratio FLOAT,
                    run_timestamp FLOAT
                )
            """)
            d = results.to_dict()
            conn.execute(
                "INSERT INTO backtest_results (strategy, asset, initial_capital, final_capital, "
                "total_return_pct, n_trades, win_rate, max_drawdown, sharpe_ratio, run_timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(d.values()),
            )
            conn.close()
        except Exception as e:
            print(f"[Backtester] KonaDB save failed (non-fatal): {e}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Kronos Backtest Runner")
    parser.add_argument("--prices",    required=True, help="JSON file with price list")
    parser.add_argument("--forecasts", required=True, help="JSON file with forecast list")
    parser.add_argument("--asset",     default="ASSET")
    parser.add_argument("--capital",   type=float, default=10000.0)
    parser.add_argument("--kona-db",   default=None, help="Path to .kona file for results")
    args = parser.parse_args()

    with open(args.prices) as f:
        prices_data = json.load(f)
    with open(args.forecasts) as f:
        forecasts_data = json.load(f)

    bt = Backtester(initial_capital=args.capital, kona_db=args.kona_db)
    comparisons = bt.compare_strategies(prices_data, forecasts_data, asset=args.asset)

    for name, result in comparisons.items():
        print(result.summary())
