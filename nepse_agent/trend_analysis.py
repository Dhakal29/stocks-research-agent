"""Technical trend analysis and momentum indicators (Chukul-style quantitative modeling)."""

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class TrendAnalysisResult:
    trend: str  # "STRONG UPTREND", "MODERATE UPTREND / RECOVERY", "NEUTRAL / SIDEWAYS", "DOWNTREND"
    score: int  # 0 to 4
    max_score: int
    summary: str
    indicators: dict[str, Any]


def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smoothed averages (Wilder's Smoothing)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def calculate_ema(prices: list[float], span: int) -> float | None:
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < span:
        return None
    multiplier = 2.0 / (span + 1.0)
    # Start with simple moving average
    ema = sum(prices[:span]) / span
    for price in prices[span:]:
        ema = (price - ema) * multiplier + ema
    return round(ema, 2)


def calculate_sma(prices: list[float], window: int) -> float | None:
    """Calculate Simple Moving Average (SMA)."""
    if len(prices) < window:
        return None
    return round(sum(prices[-window:]) / window, 2)


def evaluate_stock_trend(
    current_price: float,
    fifty_two_week_low: float | None = None,
    fifty_two_week_high: float | None = None,
    historical_closes: list[float] | None = None,
) -> TrendAnalysisResult:
    """
    Evaluates trend status using multi-factor quantitative modeling:
    1. Historical series (EMA20, SMA50, RSI 14) if historical prices are provided.
    2. 52-week position percentile & range dynamics as a reliable fallback or confirmation.
    """
    score = 0
    max_score = 0
    indicators: dict[str, Any] = {"current_price": current_price}
    signals: list[str] = []

    # 1. 52-Week Range Percentile Analysis
    if (
        fifty_two_week_low is not None
        and fifty_two_week_high is not None
        and fifty_two_week_high > fifty_two_week_low
    ):
        max_score += 1
        range_span = fifty_two_week_high - fifty_two_week_low
        position_pct = round(((current_price - fifty_two_week_low) / range_span) * 100, 1)
        indicators["52w_position_percentile"] = f"{position_pct}%"
        indicators["52w_range"] = f"NPR {fifty_two_week_low:.2f} – {fifty_two_week_high:.2f}"

        if position_pct >= 50.0:
            score += 1
            signals.append(f"Price trading in upper 52-week half ({position_pct}% of annual range)")
        else:
            signals.append(f"Price trading in lower 52-week band ({position_pct}% of annual range)")

    # 2. Historical Technical Indicators (if price history is available)
    if historical_closes and len(historical_closes) >= 15:
        # Append current price to closes if latest != current
        closes = list(historical_closes)
        if closes[-1] != current_price:
            closes.append(current_price)

        # EMA 20
        ema20 = calculate_ema(closes, 20)
        if ema20 is not None:
            max_score += 1
            indicators["ema20"] = ema20
            if current_price >= ema20:
                score += 1
                signals.append(f"Bullish: Price (NPR {current_price:.2f}) > 20 EMA (NPR {ema20:.2f})")
            else:
                signals.append(f"Bearish: Price (NPR {current_price:.2f}) < 20 EMA (NPR {ema20:.2f})")

        # SMA 50
        sma50 = calculate_sma(closes, 50)
        if sma50 is not None and ema20 is not None:
            max_score += 1
            indicators["sma50"] = sma50
            if ema20 >= sma50:
                score += 1
                signals.append(f"Golden crossover alignment: 20 EMA ({ema20:.2f}) >= 50 SMA ({sma50:.2f})")
            else:
                signals.append(f"Death cross alignment: 20 EMA ({ema20:.2f}) < 50 SMA ({sma50:.2f})")

        # RSI (14)
        rsi = calculate_rsi(closes, 14)
        if rsi is not None:
            max_score += 1
            indicators["rsi14"] = rsi
            if 50.0 <= rsi <= 70.0:
                score += 1
                signals.append(f"Positive momentum: RSI 14 is healthy ({rsi:.1f})")
            elif rsi > 70.0:
                signals.append(f"Overbought warning: RSI 14 is stretched ({rsi:.1f})")
            elif rsi < 35.0:
                signals.append(f"Oversold territory: RSI 14 indicates capitulation ({rsi:.1f})")
            else:
                signals.append(f"Subdued momentum: RSI 14 below neutral ({rsi:.1f})")

    # If no historical data was supplied, base trend on 52-week position & price momentum
    if max_score == 0:
        return TrendAnalysisResult(
            trend="DATA UNAVAILABLE",
            score=0,
            max_score=0,
            summary="Insufficient historical price data to compute quantitative technical trend.",
            indicators=indicators,
        )

    # Classify overall trend
    ratio = score / max_score
    if ratio >= 0.75:
        trend = "STRONG UPTREND"
    elif ratio >= 0.50:
        trend = "MODERATE UPTREND / RECOVERY"
    elif ratio >= 0.25:
        trend = "NEUTRAL / SIDEWAYS CONSOLIDATION"
    else:
        trend = "DOWNTREND"

    summary = f"{trend} ({score}/{max_score} indicators positive). " + "; ".join(signals) + "."
    return TrendAnalysisResult(
        trend=trend,
        score=score,
        max_score=max_score,
        summary=summary,
        indicators=indicators,
    )
