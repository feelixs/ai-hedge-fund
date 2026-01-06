"""
Pure mathematical stock scanner - NO AI/LLM calls.

Analyzes stocks using 6 categories of pure mathematical signals:
- Technical: EMA, RSI, ADX, Bollinger Bands, momentum, Hurst exponent
- Fundamentals: ROE, margins, debt ratios, valuation ratios
- Valuation: DCF, owner earnings, EV/EBITDA, residual income
- Growth: Revenue/EPS growth trends, margin expansion, PEG ratio
- Sentiment: Insider trades + news sentiment (weighted counting)
- Risk: Annualized volatility

Usage:
    poetry run python src/scanner.py
    poetry run python src/scanner.py --tickers AAPL,MSFT,GOOGL
    poetry run python src/scanner.py --detailed
    poetry run python src/scanner.py --json
"""

import argparse
import json
import os
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from src.tools.api import (
    get_prices,
    prices_to_df,
    get_financial_metrics,
    get_insider_trades,
    get_company_news,
    search_line_items,
    get_market_cap,
)

# Import calculation functions from existing agents
from src.agents.technicals import (
    calculate_trend_signals,
    calculate_mean_reversion_signals,
    calculate_momentum_signals,
    calculate_volatility_signals,
    calculate_stat_arb_signals,
    weighted_signal_combination,
)
from src.agents.valuation import (
    calculate_owner_earnings_value,
    calculate_dcf_scenarios,
    calculate_ev_ebitda_value,
    calculate_residual_income_value,
    calculate_wacc,
)
from src.agents.growth_agent import (
    analyze_growth_trends,
    analyze_valuation as analyze_growth_valuation,
    analyze_margin_trends,
    analyze_insider_conviction,
    check_financial_health,
)

# Default tickers to scan
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'INTC', 'SPY', 'QQQ']

# Intraday settings
INTRADAY_LOOKBACK_DAYS = 30  # 30 trading days of minute data
CANDLE_MINUTES = 15  # Aggregate to 15-minute candles


def aggregate_to_15min_candles(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level OHLCV data to 15-minute candles.

    Args:
        prices_df: DataFrame with 'time', 'open', 'high', 'low', 'close', 'volume' columns

    Returns:
        DataFrame with 15-minute OHLCV candles
    """
    if prices_df.empty:
        return prices_df

    df = prices_df.copy()

    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    # Resample to 15-minute candles
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample and aggregate
    df_15min = df.resample(f'{CANDLE_MINUTES}min').agg(agg_dict)

    # Drop rows with NaN (incomplete candles)
    df_15min = df_15min.dropna()

    # Reset index to have time as column again
    df_15min = df_15min.reset_index()

    return df_15min


def analyze_technical(ticker: str, start_date: str, end_date: str, api_key: str) -> dict:
    """Run technical analysis using 15-minute candles aggregated from minute data."""
    try:
        # Fetch minute-level data
        prices = get_prices(ticker, start_date, end_date, api_key, interval="minute", interval_multiplier=1)
        if not prices:
            return {"signal": "neutral", "confidence": 0, "error": "No price data"}

        # Convert to DataFrame and aggregate to 15-minute candles
        prices_df = prices_to_df(prices)
        prices_df = aggregate_to_15min_candles(prices_df)

        if prices_df.empty or len(prices_df) < 60:
            return {"signal": "neutral", "confidence": 0, "error": "Insufficient 15-min candles"}

        trend = calculate_trend_signals(prices_df)
        mean_rev = calculate_mean_reversion_signals(prices_df)
        momentum = calculate_momentum_signals(prices_df)
        volatility = calculate_volatility_signals(prices_df)
        stat_arb = calculate_stat_arb_signals(prices_df)

        combined = weighted_signal_combination(
            {
                "trend": trend,
                "mean_reversion": mean_rev,
                "momentum": momentum,
                "volatility": volatility,
                "stat_arb": stat_arb,
            },
            {
                "trend": 0.25,
                "mean_reversion": 0.20,
                "momentum": 0.25,
                "volatility": 0.15,
                "stat_arb": 0.15,
            },
        )

        # Return full reasoning with all metrics for day trading analysis
        return {
            "signal": combined["signal"],
            "confidence": round(combined["confidence"] * 100),
            "reasoning": {
                "trend_following": {
                    "signal": trend["signal"],
                    "confidence": round(trend["confidence"] * 100),
                    "metrics": trend.get("metrics", {}),
                },
                "mean_reversion": {
                    "signal": mean_rev["signal"],
                    "confidence": round(mean_rev["confidence"] * 100),
                    "metrics": mean_rev.get("metrics", {}),
                },
                "momentum": {
                    "signal": momentum["signal"],
                    "confidence": round(momentum["confidence"] * 100),
                    "metrics": momentum.get("metrics", {}),
                },
                "volatility": {
                    "signal": volatility["signal"],
                    "confidence": round(volatility["confidence"] * 100),
                    "metrics": volatility.get("metrics", {}),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb["signal"],
                    "confidence": round(stat_arb["confidence"] * 100),
                    "metrics": stat_arb.get("metrics", {}),
                },
            },
        }
    except Exception as e:
        return {"signal": "neutral", "confidence": 0, "error": str(e)}


def analyze_fundamentals(ticker: str, end_date: str, api_key: str) -> dict:
    """Run fundamentals analysis using threshold-based scoring."""
    try:
        metrics = get_financial_metrics(ticker, end_date, "ttm", 10, api_key)
        if not metrics:
            return {"signal": "neutral", "confidence": 0, "error": "No metrics data"}

        m = metrics[0]

        # Profitability score
        profitability = sum([
            m.return_on_equity is not None and m.return_on_equity > 0.15,
            m.net_margin is not None and m.net_margin > 0.20,
            m.operating_margin is not None and m.operating_margin > 0.15,
        ])

        # Growth score
        growth = sum([
            m.revenue_growth is not None and m.revenue_growth > 0.10,
            m.earnings_growth is not None and m.earnings_growth > 0.10,
            m.book_value_growth is not None and m.book_value_growth > 0.10,
        ])

        # Health score
        health = sum([
            m.current_ratio is not None and m.current_ratio > 1.5,
            m.debt_to_equity is not None and m.debt_to_equity < 0.5,
            (m.free_cash_flow_per_share is not None and m.earnings_per_share is not None and
             m.free_cash_flow_per_share > m.earnings_per_share * 0.8),
        ])

        # Valuation (inverted - high ratios = bearish)
        valuation_bearish = sum([
            m.price_to_earnings_ratio is not None and m.price_to_earnings_ratio > 25,
            m.price_to_book_ratio is not None and m.price_to_book_ratio > 3,
            m.price_to_sales_ratio is not None and m.price_to_sales_ratio > 5,
        ])

        bullish = profitability + growth + health
        bearish = valuation_bearish

        if bullish > bearish:
            signal = "bullish"
        elif bearish > bullish:
            signal = "bearish"
        else:
            signal = "neutral"

        total = max(bullish + bearish, 1)
        confidence = round(max(bullish, bearish) / total * 100)

        # Return with raw metrics for comprehensive analysis
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "return_on_equity": m.return_on_equity,
                "net_margin": m.net_margin,
                "operating_margin": m.operating_margin,
                "revenue_growth": m.revenue_growth,
                "earnings_growth": m.earnings_growth,
                "book_value_growth": m.book_value_growth,
                "current_ratio": m.current_ratio,
                "debt_to_equity": m.debt_to_equity,
                "price_to_earnings": m.price_to_earnings_ratio,
                "price_to_book": m.price_to_book_ratio,
                "price_to_sales": m.price_to_sales_ratio,
                "free_cash_flow_per_share": m.free_cash_flow_per_share,
                "earnings_per_share": m.earnings_per_share,
            },
            "scores": {
                "profitability": profitability,
                "growth": growth,
                "health": health,
                "valuation_bearish": valuation_bearish,
            },
        }
    except Exception as e:
        return {"signal": "neutral", "confidence": 0, "error": str(e)}


def analyze_valuation_signal(ticker: str, end_date: str, api_key: str) -> dict:
    """Run valuation analysis using DCF and comparable methods."""
    try:
        # Get financial metrics
        financial_metrics = get_financial_metrics(ticker, end_date, "ttm", 8, api_key)
        if not financial_metrics:
            return {"signal": "neutral", "confidence": 0, "error": "No metrics data"}

        most_recent_metrics = financial_metrics[0]

        # Get line items for DCF
        line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow", "net_income", "depreciation_and_amortization",
                "capital_expenditure", "working_capital", "total_debt",
                "cash_and_equivalents", "interest_expense", "revenue",
                "operating_income", "ebit", "ebitda"
            ],
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )

        if len(line_items) < 2:
            return {"signal": "neutral", "confidence": 0, "error": "Insufficient line items"}

        li_curr, li_prev = line_items[0], line_items[1]

        # Working capital change
        if li_curr.working_capital is not None and li_prev.working_capital is not None:
            wc_change = li_curr.working_capital - li_prev.working_capital
        else:
            wc_change = 0

        # Owner Earnings Value
        owner_val = calculate_owner_earnings_value(
            net_income=li_curr.net_income,
            depreciation=li_curr.depreciation_and_amortization,
            capex=li_curr.capital_expenditure,
            working_capital_change=wc_change,
            growth_rate=most_recent_metrics.earnings_growth or 0.05,
        )

        # WACC calculation
        wacc = calculate_wacc(
            market_cap=most_recent_metrics.market_cap or 0,
            total_debt=getattr(li_curr, 'total_debt', None),
            cash=getattr(li_curr, 'cash_and_equivalents', None),
            interest_coverage=most_recent_metrics.interest_coverage,
            debt_to_equity=most_recent_metrics.debt_to_equity,
        )

        # FCF history for DCF
        fcf_history = []
        for li in line_items:
            if hasattr(li, 'free_cash_flow') and li.free_cash_flow is not None:
                fcf_history.append(li.free_cash_flow)

        # DCF with scenarios
        dcf_results = calculate_dcf_scenarios(
            fcf_history=fcf_history,
            growth_metrics={
                'revenue_growth': most_recent_metrics.revenue_growth,
                'fcf_growth': most_recent_metrics.free_cash_flow_growth,
                'earnings_growth': most_recent_metrics.earnings_growth
            },
            wacc=wacc,
            market_cap=most_recent_metrics.market_cap or 0,
            revenue_growth=most_recent_metrics.revenue_growth
        )
        dcf_val = dcf_results['expected_value']

        # EV/EBITDA value
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics)

        # Residual Income Value
        rim_val = calculate_residual_income_value(
            market_cap=most_recent_metrics.market_cap,
            net_income=li_curr.net_income,
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=most_recent_metrics.book_value_growth or 0.03,
        )

        # Get market cap
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        if not market_cap:
            return {"signal": "neutral", "confidence": 0, "error": "No market cap"}

        # Aggregate valuation methods
        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        total_weight = sum(v["weight"] for v in method_values.values() if v["value"] > 0)
        if total_weight == 0:
            return {"signal": "neutral", "confidence": 0, "error": "All valuations zero"}

        for v in method_values.values():
            v["gap"] = (v["value"] - market_cap) / market_cap if v["value"] > 0 else None

        weighted_gap = sum(
            v["weight"] * v["gap"] for v in method_values.values() if v["gap"] is not None
        ) / total_weight

        # Calculate weighted intrinsic value
        weighted_intrinsic_value = sum(
            v["weight"] * v["value"] for v in method_values.values() if v["value"] > 0
        ) / total_weight

        signal = "bullish" if weighted_gap > 0.15 else "bearish" if weighted_gap < -0.15 else "neutral"
        confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100))

        # Return with detailed valuation breakdowns
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "market_cap": market_cap,
                "dcf_value": dcf_val,
                "owner_earnings_value": owner_val,
                "ev_ebitda_value": ev_ebitda_val,
                "residual_income_value": rim_val,
                "weighted_intrinsic_value": weighted_intrinsic_value,
                "valuation_gap_pct": round(weighted_gap * 100, 1),
                "wacc": wacc,
            },
        }
    except Exception as e:
        return {"signal": "neutral", "confidence": 0, "error": str(e)}


def analyze_growth_signal(ticker: str, end_date: str, api_key: str) -> dict:
    """Run growth analysis using trend-based scoring."""
    try:
        # Get financial metrics
        financial_metrics = get_financial_metrics(ticker, end_date, "ttm", 12, api_key)
        if not financial_metrics or len(financial_metrics) < 4:
            return {"signal": "neutral", "confidence": 0, "error": "Insufficient metrics"}

        most_recent_metrics = financial_metrics[0]

        # Get insider trades
        insider_trades = get_insider_trades(ticker, end_date, limit=1000, api_key=api_key)

        # Run growth analysis components
        growth_trends = analyze_growth_trends(financial_metrics)
        valuation_metrics = analyze_growth_valuation(most_recent_metrics)
        margin_trends = analyze_margin_trends(financial_metrics)
        insider_conviction = analyze_insider_conviction(insider_trades)
        financial_health = check_financial_health(most_recent_metrics)

        # Calculate weighted score
        scores = {
            "growth": growth_trends['score'],
            "valuation": valuation_metrics['score'],
            "margins": margin_trends['score'],
            "insider": insider_conviction['score'],
            "health": financial_health['score']
        }

        weights = {
            "growth": 0.40,
            "valuation": 0.25,
            "margins": 0.15,
            "insider": 0.10,
            "health": 0.10
        }

        weighted_score = sum(scores[key] * weights[key] for key in scores)

        if weighted_score > 0.6:
            signal = "bullish"
        elif weighted_score < 0.4:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = round(abs(weighted_score - 0.5) * 2 * 100)

        # Count insider buys/sells
        insider_buys = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares > 0)
        insider_sells = sum(1 for t in insider_trades if t.transaction_shares and t.transaction_shares < 0)

        # Return with detailed growth metrics
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "revenue_growth": most_recent_metrics.revenue_growth,
                "earnings_growth": most_recent_metrics.earnings_growth,
                "fcf_growth": most_recent_metrics.free_cash_flow_growth,
                "operating_margin": most_recent_metrics.operating_margin,
                "gross_margin": most_recent_metrics.gross_margin,
                "peg_ratio": most_recent_metrics.peg_ratio,
                "insider_buy_count": insider_buys,
                "insider_sell_count": insider_sells,
            },
            "component_scores": scores,
        }
    except Exception as e:
        return {"signal": "neutral", "confidence": 0, "error": str(e)}


def analyze_sentiment_signal(ticker: str, end_date: str, api_key: str) -> dict:
    """Run sentiment analysis using weighted insider + news counting."""
    try:
        # Get insider trades
        insider_trades = get_insider_trades(ticker, end_date, limit=1000, api_key=api_key)

        # Get signals from insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        # Get company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get sentiment from news (the API returns sentiment field)
        if company_news:
            sentiment = pd.Series([n.sentiment for n in company_news if n.sentiment]).dropna()
            news_signals = np.where(
                sentiment == "negative", "bearish",
                np.where(sentiment == "positive", "bullish", "neutral")
            ).tolist()
        else:
            news_signals = []

        # Weights
        insider_weight = 0.3
        news_weight = 0.7

        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            signal = "bullish"
        elif bearish_signals > bullish_signals:
            signal = "bearish"
        else:
            signal = "neutral"

        # Calculate confidence
        total_weighted = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        if total_weighted > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted) * 100)
        else:
            confidence = 0

        # Count news by sentiment
        news_positive = news_signals.count("bullish") if news_signals else 0
        news_negative = news_signals.count("bearish") if news_signals else 0
        news_neutral = news_signals.count("neutral") if news_signals else 0

        # Determine sub-signals
        insider_net = "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else (
            "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral"
        )
        news_net = "bullish" if news_positive > news_negative else (
            "bearish" if news_negative > news_positive else "neutral"
        )

        # Return with detailed sentiment metrics
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "insider_buys": insider_signals.count("bullish"),
                "insider_sells": insider_signals.count("bearish"),
                "news_positive": news_positive,
                "news_negative": news_negative,
                "news_neutral": news_neutral,
                "insider_net_signal": insider_net,
                "news_net_signal": news_net,
            },
        }
    except Exception as e:
        return {"signal": "neutral", "confidence": 0, "error": str(e)}


def analyze_risk(ticker: str, start_date: str, end_date: str, api_key: str) -> dict:
    """Calculate risk metrics using 15-minute candle volatility."""
    try:
        # Fetch minute data and aggregate to 15-min candles
        prices = get_prices(ticker, start_date, end_date, api_key, interval="minute", interval_multiplier=1)
        if not prices:
            return {"risk_level": "UNKNOWN", "volatility": 0, "error": "No price data"}

        prices_df = prices_to_df(prices)
        prices_df = aggregate_to_15min_candles(prices_df)

        if prices_df.empty:
            return {"risk_level": "UNKNOWN", "volatility": 0, "error": "No 15-min candles"}

        returns = prices_df["close"].pct_change().dropna()

        if len(returns) < 50:
            return {"risk_level": "UNKNOWN", "volatility": 0, "error": "Insufficient data"}

        # Annualize 15-minute volatility
        # 26 bars per trading day (6.5 hours / 0.25 hours), 252 trading days
        bars_per_year = 26 * 252  # 6552
        bar_vol = returns.std()
        annual_vol = bar_vol * math.sqrt(bars_per_year)

        if annual_vol < 0.20:
            risk_level = "LOW"
        elif annual_vol < 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {"risk_level": risk_level, "volatility": round(annual_vol * 100, 1)}
    except Exception as e:
        return {"risk_level": "UNKNOWN", "volatility": 0, "error": str(e)}


def calculate_overall_signal(signals: dict) -> str:
    """Calculate overall signal from all categories (equal weights)."""
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    # Equal weights for 5 signal categories (risk is informational only)
    categories = ["technical", "fundamentals", "valuation", "growth", "sentiment"]
    total = 0
    for cat in categories:
        if cat in signals and "signal" in signals[cat]:
            total += signal_values.get(signals[cat]["signal"], 0)

    if total >= 2:
        return "BULLISH"
    elif total <= -2:
        return "BEARISH"
    else:
        return "NEUTRAL"


def format_signal(signal: str, detailed: bool = False, confidence: int = 0) -> str:
    """Format signal for display."""
    signal_upper = signal.upper()
    if detailed and confidence > 0:
        return f"{signal_upper}({confidence}%)"
    return signal_upper


def print_table(results: list[dict], detailed: bool = False):
    """Print results as a formatted table."""
    # Header
    print("=" * 100)
    print("                              STOCK SCANNER RESULTS")
    print(f"                              {datetime.now().strftime('%Y-%m-%d')} (15-min candles, 30-day lookback)")
    print("=" * 100)
    print()

    if detailed:
        header = f"{'Ticker':<8}| {'Technical':<14}| {'Fundmntl':<14}| {'Valuatn':<14}| {'Growth':<14}| {'Sentmnt':<14}| {'Risk':<8}| {'Overall':<10}"
        separator = "-" * 8 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 8 + "|" + "-" * 10
    else:
        header = f"{'Ticker':<8}| {'Technical':<10}| {'Fundmntl':<10}| {'Valuatn':<10}| {'Growth':<10}| {'Sentmnt':<10}| {'Risk':<8}| {'Overall':<10}"
        separator = "-" * 8 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 10

    print(header)
    print(separator)

    for r in results:
        ticker = r["ticker"]
        signals = r["signals"]

        tech = format_signal(signals.get("technical", {}).get("signal", "N/A"), detailed, signals.get("technical", {}).get("confidence", 0))
        fund = format_signal(signals.get("fundamentals", {}).get("signal", "N/A"), detailed, signals.get("fundamentals", {}).get("confidence", 0))
        val = format_signal(signals.get("valuation", {}).get("signal", "N/A"), detailed, signals.get("valuation", {}).get("confidence", 0))
        growth = format_signal(signals.get("growth", {}).get("signal", "N/A"), detailed, signals.get("growth", {}).get("confidence", 0))
        sent = format_signal(signals.get("sentiment", {}).get("signal", "N/A"), detailed, signals.get("sentiment", {}).get("confidence", 0))
        risk = signals.get("risk", {}).get("risk_level", "N/A")
        overall = r["overall"]

        if detailed:
            print(f"{ticker:<8}| {tech:<14}| {fund:<14}| {val:<14}| {growth:<14}| {sent:<14}| {risk:<8}| {overall:<10}")
        else:
            print(f"{ticker:<8}| {tech:<10}| {fund:<10}| {val:<10}| {growth:<10}| {sent:<10}| {risk:<8}| {overall:<10}")

    print()
    print("Legend: Risk levels - LOW (<20% annual vol), MEDIUM (20-35%), HIGH (>35%)")
    if detailed:
        print("        Confidence shown in parentheses")


def run_scanner(tickers: list[str], api_key: str, detailed: bool = False, output_json: bool = False) -> list[dict]:
    """Run the scanner on all tickers."""
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Intraday lookback for technical analysis (30 trading days of minute data)
    intraday_start_date = (datetime.now() - timedelta(days=INTRADAY_LOOKBACK_DAYS + 15)).strftime("%Y-%m-%d")  # +15 for weekends/holidays

    # Longer lookback for fundamental/risk analysis
    fundamental_start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    results = []

    for ticker in tickers:
        print(f"Scanning {ticker}...", end=" ", flush=True)

        signals = {}

        # Technical analysis (using 15-minute candles from minute data)
        signals["technical"] = analyze_technical(ticker, intraday_start_date, end_date, api_key)

        # Fundamentals analysis
        signals["fundamentals"] = analyze_fundamentals(ticker, end_date, api_key)

        # Valuation analysis
        signals["valuation"] = analyze_valuation_signal(ticker, end_date, api_key)

        # Growth analysis
        signals["growth"] = analyze_growth_signal(ticker, end_date, api_key)

        # Sentiment analysis
        signals["sentiment"] = analyze_sentiment_signal(ticker, end_date, api_key)

        # Risk analysis (use intraday data for recent volatility)
        signals["risk"] = analyze_risk(ticker, intraday_start_date, end_date, api_key)

        # Overall signal
        overall = calculate_overall_signal(signals)

        results.append({
            "ticker": ticker,
            "signals": signals,
            "overall": overall,
        })

        print("Done")

    print()

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results, detailed)

    return results


def get_instructions_content() -> str:
    """Generate the INSTRUCTIONS.md content for agents."""
    return """# Stock Scanner Results - Day Trading & Momentum Guide

## Timeframe: 15-Minute Candles

**This scanner uses 15-MINUTE price bars aggregated from minute data (30-day lookback).**

- Optimized for **day trading** and **intraday momentum plays**
- ~780 fifteen-minute candles per scan (30 trading days × 26 bars/day)
- Technical indicators (RSI, momentum, EMA) calculated on 15-min timeframe
- Ideal for entries/exits within hours to days

---

## Quick Reference: Key Technical Metrics

### Momentum Indicators (Primary for Trading Decisions)

| Metric | Location | Bullish | Bearish | Use |
|--------|----------|---------|---------|-----|
| `momentum_1m` | technical.reasoning.momentum.metrics | > 0.03 | < -0.03 | **Primary entry signal** (21-bar momentum) |
| `volume_momentum` | technical.reasoning.momentum.metrics | > 1.2 | N/A | **Confirmation required** |
| `rsi_14` | technical.reasoning.mean_reversion.metrics | < 30 (oversold) | > 70 (overbought) | Reversal signals |
| `z_score` | technical.reasoning.mean_reversion.metrics | < -2 | > +2 | Mean reversion plays |

### Trend Indicators

| Metric | Location | Bullish | Bearish | Use |
|--------|----------|---------|---------|-----|
| `adx` | technical.reasoning.trend_following.metrics | > 25 (strong) | > 25 (strong) | Trend strength |
| `trend_strength` | technical.reasoning.trend_following.metrics | > 0.25 | > 0.25 | Confidence multiplier |

### Volatility & Risk

| Metric | Location | Low Risk | High Risk | Use |
|--------|----------|----------|-----------|-----|
| `atr_ratio` | technical.reasoning.volatility.metrics | < 0.02 | > 0.04 | Stop loss sizing (2x ATR) |
| `volatility_regime` | technical.reasoning.volatility.metrics | < 0.8 | > 1.2 | Position sizing |
| `historical_volatility` | technical.reasoning.volatility.metrics | < 0.20 | > 0.35 | Annual vol % |

---

## Momentum Play Identification (Intraday)

### STRONG BUY Setup (Momentum Breakout)
All conditions must be true:
1. `momentum_1m` > 0.03 (3%+ over 21 fifteen-minute bars ≈ 5 hours)
2. `volume_momentum` > 1.2 (volume confirmation)
3. `rsi_14` < 70 (not overbought)
4. `adx` > 25 (trend is strong)
5. Technical signal = "bullish"

### BUY Setup (Momentum Continuation)
Most conditions true:
1. `momentum_1m` > 0.02 (2%+ short-term momentum)
2. `volume_momentum` > 1.0
3. `rsi_14` between 40-65
4. Trend following signal = "bullish"

### MEAN REVERSION Long (Oversold Bounce)
1. `z_score` < -2 (statistically oversold on 15-min chart)
2. `rsi_14` < 30
3. `price_vs_bb` < 0.2 (near lower Bollinger Band)
4. Wait for reversal candle confirmation before entry

### AVOID / SHORT Setup
1. `rsi_14` > 70 (overbought)
2. `momentum_1m` < -0.03 (negative momentum)
3. `volume_momentum` > 1.2 on down moves
4. OR `volatility_regime` > 1.3 (elevated volatility)

---

## Risk Management Rules

### Position Sizing by Volatility Regime
| volatility_regime | Position Size |
|-------------------|---------------|
| < 0.8 | Full size (can go aggressive) |
| 0.8 - 1.0 | Normal size |
| 1.0 - 1.2 | Reduce by 25% |
| > 1.2 | Reduce by 50% or avoid |

### Stop Loss Calculation
```
Stop Loss = Entry Price - (2 × atr_ratio × Entry Price)
```
Example: Entry at $100, atr_ratio = 0.025
- Stop Loss = $100 - (2 × 0.025 × $100) = $95

### Risk Level Guidelines
| Risk Level | Annual Volatility | Max Position % |
|------------|-------------------|----------------|
| LOW | < 20% | 20% of portfolio |
| MEDIUM | 20-35% | 10% of portfolio |
| HIGH | > 35% | 5% of portfolio |

---

## Signal Categories Reference

### Technical Analysis (Most Important for Momentum)
Contains 5 sub-strategies with detailed metrics:
- **trend_following**: EMA alignment, ADX strength
- **mean_reversion**: RSI, Z-score, Bollinger position
- **momentum**: 1m/3m/6m returns, volume momentum
- **volatility**: Historical vol, ATR, regime
- **statistical_arbitrage**: Hurst exponent, skewness

### Fundamentals Analysis
Background context (less important for short-term trades):
- `metrics.return_on_equity`, `net_margin`, `operating_margin`
- `metrics.revenue_growth`, `earnings_growth`
- `metrics.price_to_earnings`, `price_to_book`

### Valuation Analysis
For confluence with technicals:
- `metrics.valuation_gap_pct`: Positive = undervalued, Negative = overvalued
- Undervalued + bullish momentum = stronger conviction

### Growth Analysis
Confirms momentum sustainability:
- `metrics.revenue_growth`, `earnings_growth`, `fcf_growth`
- `metrics.insider_buy_count` vs `insider_sell_count`

### Sentiment Analysis
Confirms market bias:
- `metrics.news_positive` vs `news_negative` count
- `metrics.insider_net_signal`: Are insiders buying?

---

## Expected Output Format

After analyzing all ticker files, provide recommendations:

```markdown
## Trading Recommendations

| Ticker | Action | Entry Signal | Risk | Key Metrics |
|--------|--------|--------------|------|-------------|
| NVDA | BUY | Momentum breakout | MED | Mom_1m: +8%, RSI: 58, Vol: 1.4x, ADX: 32 |
| AMD | BUY | Trend continuation | MED | Mom_1m: +5%, RSI: 52, bullish EMA |
| TSLA | AVOID | Overbought | HIGH | RSI: 78, Vol_regime: 1.4 |
| INTC | SHORT | Momentum breakdown | MED | Mom_1m: -7%, Vol: 1.3x, RSI: 35 falling |

## Position Sizing
Based on volatility regime and risk levels:
- NVDA: 10% position (MED risk, vol_regime: 1.0)
- AMD: 10% position (MED risk)

## Stop Losses
- NVDA: $XXX (2x ATR below entry)
- AMD: $XXX (2x ATR below entry)

## Watchlist (Not Ready Yet)
- AAPL: Waiting for volume confirmation
- GOOGL: RSI still elevated, wait for pullback
```

---

## File Structure
- `INSTRUCTIONS.md` - This file (trading guide)
- `{TICKER}.json` - Full analysis data per ticker
"""


def save_scan_results(results: list[dict], scan_date: str) -> str:
    """Save scan results to timestamped directory.

    Returns the path to the created directory.
    """
    # Create timestamped directory relative to project root (parent of src/)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "scans", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save individual ticker files
    for result in results:
        ticker = result["ticker"]
        ticker_data = {
            "ticker": ticker,
            "scan_date": scan_date,
            "signals": result["signals"],
            "overall": result["overall"],
        }

        ticker_file = os.path.join(output_dir, f"{ticker}.json")
        with open(ticker_file, "w") as f:
            json.dump(ticker_data, f, indent=2)

    # Save INSTRUCTIONS.md
    instructions_file = os.path.join(output_dir, "INSTRUCTIONS.md")
    with open(instructions_file, "w") as f:
        f.write(get_instructions_content())

    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pure mathematical stock scanner - NO AI/LLM calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python src/scanner.py
  poetry run python src/scanner.py --tickers AAPL,MSFT,GOOGL
  poetry run python src/scanner.py --detailed
  poetry run python src/scanner.py --json
        """
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=f"Comma-separated list of tickers (default: {','.join(DEFAULT_TICKERS)})"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show confidence percentages for each signal"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")

    if not api_key:
        print("Error: FINANCIAL_DATASETS_API_KEY not found in .env file")
        return

    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    # Run scanner
    results = run_scanner(tickers, api_key, args.detailed, args.json)

    # Save results to timestamped directory
    scan_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = save_scan_results(results, scan_date)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
