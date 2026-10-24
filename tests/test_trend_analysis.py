"""Unit tests for Chukul-style technical trend analysis."""

from nepse_agent.trend_analysis import (
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    evaluate_stock_trend,
)


def test_rsi_calculation():
    # 15 flat prices => RSI = 50.0
    flat_prices = [100.0] * 16
    assert calculate_rsi(flat_prices, 14) == 50.0

    # Consistently increasing prices => RSI = 100.0
    rising_prices = [100.0 + i * 2.0 for i in range(20)]
    assert calculate_rsi(rising_prices, 14) == 100.0


def test_ema_and_sma():
    prices = list(range(1, 60))
    sma50 = calculate_sma(prices, 50)
    ema20 = calculate_ema(prices, 20)
    assert sma50 is not None
    assert ema20 is not None
    assert ema20 > sma50  # Steady upward trend has EMA20 > SMA50


def test_evaluate_stock_trend_52w():
    # Price near 52w low (340 in 337 - 571 range) -> Downtrend / Lower band
    res_low = evaluate_stock_trend(current_price=340.0, fifty_two_week_low=337.0, fifty_two_week_high=571.0)
    assert res_low.trend == "DOWNTREND"
    assert res_low.score == 0

    # Price in upper half (500 in 337 - 571 range) -> Moderate recovery/uptrend
    res_high = evaluate_stock_trend(current_price=500.0, fifty_two_week_low=337.0, fifty_two_week_high=571.0)
    assert res_high.trend == "STRONG UPTREND"
    assert res_high.score == 1


def test_evaluate_stock_trend_with_series():
    # Steady upward series with high RSI and EMA > SMA
    series = [100.0 + i * 1.5 for i in range(55)]
    res = evaluate_stock_trend(
        current_price=series[-1],
        fifty_two_week_low=100.0,
        fifty_two_week_high=series[-1],
        historical_closes=series,
    )
    assert "UPTREND" in res.trend
    assert res.max_score >= 3
