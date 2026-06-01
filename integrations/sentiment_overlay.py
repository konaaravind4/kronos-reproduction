"""
Sentiment Overlay for Kronos Financial Forecasting
===================================================
Pulls real-time market sentiment from the Real-time-Sentiment-Intelligence-Dashboard
and blends it with Kronos price forecasts for sentiment-enhanced predictions.

Usage:
    from integrations.sentiment_overlay import SentimentOverlay

    overlay = SentimentOverlay(sentiment_api="http://localhost:8000")
    mood = overlay.get_market_mood("BTC", hours=1)
    adjusted = overlay.adjust_forecast(kronos_forecast=44000.0, ticker="BTC")
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class MarketMood:
    ticker: str
    signal: str          # bullish | bearish | neutral
    score: float         # -1.0 to +1.0
    confidence: float    # 0.0 to 1.0
    bull_bear_ratio: float
    total_signals: int
    timestamp: float


class SentimentOverlay:
    """
    Integrates market sentiment signals into Kronos price forecasts.
    
    Fetches from Real-time-Sentiment-Intelligence-Dashboard's /sentiment/financial/summary
    endpoint, then adjusts Kronos forecasts using a configurable blend weight.
    
    Falls back to neutral sentiment if the dashboard is unreachable.
    """

    def __init__(
        self,
        sentiment_api: str = "http://localhost:8000",
        timeout_s: float = 3.0,
    ):
        self.api = sentiment_api.rstrip("/")
        self.timeout = timeout_s
        self._cache: dict[str, tuple[float, MarketMood]] = {}  # ticker → (ts, mood)
        self._cache_ttl = 60.0  # cache for 60 seconds

    def get_market_mood(self, ticker: str = "BTC", hours: float = 1.0) -> MarketMood:
        """
        Fetch current market sentiment for a ticker.

        Args:
            ticker: Asset symbol (BTC, ETH, AAPL, etc.)
            hours: Time window for sentiment aggregation.

        Returns:
            MarketMood with signal, score, and confidence.
        """
        # Cache check
        cached = self._cache.get(ticker)
        if cached and (time.time() - cached[0]) < self._cache_ttl:
            return cached[1]

        try:
            url = f"{self.api}/sentiment/financial/summary?hours={hours}&ticker={ticker}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())

            mood = MarketMood(
                ticker=ticker,
                signal=data.get("dominant", "neutral"),
                score=data.get("avg_score", 0.0),
                confidence=min(abs(data.get("avg_score", 0.0)) * 2, 1.0),
                bull_bear_ratio=data.get("bull_bear_ratio", 1.0),
                total_signals=data.get("total", 0),
                timestamp=time.time(),
            )
        except Exception:
            # Fallback: neutral mood
            mood = MarketMood(
                ticker=ticker,
                signal="neutral",
                score=0.0,
                confidence=0.0,
                bull_bear_ratio=1.0,
                total_signals=0,
                timestamp=time.time(),
            )

        self._cache[ticker] = (time.time(), mood)
        return mood

    def adjust_forecast(
        self,
        kronos_forecast: float,
        ticker: str = "BTC",
        blend_weight: float = 0.15,
        hours: float = 1.0,
    ) -> float:
        """
        Adjust a Kronos price forecast using market sentiment.

        The sentiment score nudges the forecast up (bullish) or down (bearish).
        A blend_weight of 0.15 means sentiment contributes 15% of the adjustment.

        Args:
            kronos_forecast: Kronos raw price forecast.
            ticker: Asset ticker for sentiment lookup.
            blend_weight: Sentiment influence (0.0 = Kronos only, 1.0 = sentiment only).
            hours: Sentiment lookback window.

        Returns:
            Sentiment-adjusted forecast price.

        Example:
            >>> overlay = SentimentOverlay()
            >>> raw = 44000.0   # Kronos forecast
            >>> adjusted = overlay.adjust_forecast(raw, ticker="BTC", blend_weight=0.2)
            >>> print(f"Raw: {raw} → Adjusted: {adjusted:.2f}")
        """
        mood = self.get_market_mood(ticker, hours=hours)

        # Sentiment adjustment: +/- % based on score × blend_weight
        sentiment_adjustment = mood.score * blend_weight
        adjusted = kronos_forecast * (1 + sentiment_adjustment)
        return round(adjusted, 2)

    def batch_adjust(
        self,
        forecasts: list[float],
        tickers: list[str] | None = None,
        blend_weight: float = 0.15,
    ) -> list[float]:
        """
        Adjust a series of Kronos forecasts with sentiment.

        Args:
            forecasts: List of Kronos price forecasts.
            tickers: Corresponding asset tickers. If None, uses "MARKET" for all.
            blend_weight: Sentiment blend weight.

        Returns:
            List of adjusted forecasts.
        """
        if tickers is None:
            tickers = ["MARKET"] * len(forecasts)
        return [
            self.adjust_forecast(f, t, blend_weight)
            for f, t in zip(forecasts, tickers)
        ]

    def sentiment_report(self, tickers: list[str] | None = None) -> str:
        """
        Generate a human-readable market sentiment report.

        Args:
            tickers: List of tickers to report on.

        Returns:
            Formatted text report.
        """
        tickers = tickers or ["BTC", "ETH", "SPY", "AAPL", "NVDA"]
        lines = ["=" * 55, "  Market Sentiment Report", "=" * 55]
        for ticker in tickers:
            mood = self.get_market_mood(ticker)
            emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(mood.signal, "➡️")
            lines.append(
                f"  {ticker:<8} {emoji} {mood.signal.upper():<8} "
                f"score={mood.score:+.3f}  confidence={mood.confidence:.0%}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)
