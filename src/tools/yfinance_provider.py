"""yfinance-backed implementations of the data functions in ``src/tools/api.py``.

Free, no-key data source (Yahoo Finance) for arbitrary tickers. Each function
mirrors the signature of its ``api.py`` counterpart and returns the same model
objects, so agents are unchanged. Selected via ``DATA_SOURCE=yfinance`` (see
``src/tools/data_source.py``); ``api.py`` routes to here.

Fidelity is lower than financialdatasets (see
``docs/superpowers/specs/2026-06-08-yfinance-data-source-design.md``):
financial metrics are a single current snapshot, history is ~4-5 annual periods,
news has no sentiment, insider trades are partial. Every model field is optional,
so agents run; only signal quality drops.

``import yfinance`` is a hard dependency — if it is missing the import raises
(no silent fallback, per project policy). Genuine *data* failures return an empty
list / ``None`` (the existing ``api.py`` contract agents already tolerate) and
print a warning to stderr.
"""

import math
import sys
from datetime import datetime, timedelta

import yfinance as yf

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price

# Per-ticker memoization of raw Yahoo fetches so the many metric/line-item calls
# in a single run don't re-hit Yahoo and trip rate limits.
_INFO_CACHE: dict[str, dict] = {}
_STMTS_CACHE: dict[str, tuple] = {}
_NEWS_CACHE: dict[str, list] = {}
_INSIDER_CACHE: dict[str, object] = {}


def _warn(msg: str) -> None:
    print(f"[yfinance] {msg}", file=sys.stderr)


def _info(ticker: str) -> dict:
    if ticker not in _INFO_CACHE:
        try:
            _INFO_CACHE[ticker] = yf.Ticker(ticker).get_info() or {}
        except Exception as e:  # noqa: BLE001
            _warn(f"get_info({ticker}) failed: {e}")
            _INFO_CACHE[ticker] = {}
    return _INFO_CACHE[ticker]


def _stmts(ticker: str) -> tuple:
    """Return (income_stmt, balance_sheet, cashflow) DataFrames (or None each)."""
    if ticker not in _STMTS_CACHE:
        tk = yf.Ticker(ticker)
        out = []
        for name in ("income_stmt", "balance_sheet", "cashflow"):
            try:
                df = getattr(tk, name)
                out.append(df if df is not None and not df.empty else None)
            except Exception as e:  # noqa: BLE001
                _warn(f"{name}({ticker}) failed: {e}")
                out.append(None)
        _STMTS_CACHE[ticker] = tuple(out)
    return _STMTS_CACHE[ticker]


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _cell(df, col, labels: list[str]) -> float | None:
    if df is None or col not in df.columns:
        return None
    for label in labels:
        if label in df.index:
            v = _num(df.loc[label, col])
            if v is not None:
                return v
    return None


def _interval(interval: str) -> str:
    return {"minute": "1m", "day": "1d", "week": "1wk", "month": "1mo", "year": "3mo"}.get(interval, "1d")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None, interval: str = "day", interval_multiplier: int = 1) -> list[Price]:
    ticker = ticker.upper()
    try:
        # yfinance treats `end` as exclusive; bump it a day to include end_date.
        end_plus = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start_date, end=end_plus, interval=_interval(interval), auto_adjust=False)
    except Exception as e:  # noqa: BLE001
        _warn(f"history({ticker}) failed: {e}")
        return []
    if df is None or df.empty:
        return []

    prices: list[Price] = []
    for ts, row in df.iterrows():
        prices.append(
            Price(
                open=_num(row.get("Open")) or 0.0,
                close=_num(row.get("Close")) or 0.0,
                high=_num(row.get("High")) or 0.0,
                low=_num(row.get("Low")) or 0.0,
                volume=int(_num(row.get("Volume")) or 0),
                time=ts.strftime("%Y-%m-%d"),
            )
        )
    return prices


# ---------------------------------------------------------------------------
# Market cap
# ---------------------------------------------------------------------------
def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> float | None:
    ticker = ticker.upper()
    info = _info(ticker)
    shares = _num(info.get("sharesOutstanding"))

    # Prefer shares x close(end_date) so backtest dates get a date-appropriate
    # value; fall back to the current marketCap from info.
    if shares:
        prices = get_prices(ticker, (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d"), end_date)
        if prices:
            return shares * prices[-1].close
    return _num(info.get("marketCap"))


# ---------------------------------------------------------------------------
# Financial metrics (single current snapshot mapped from .info)
# ---------------------------------------------------------------------------
def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str = None) -> list[FinancialMetrics]:
    ticker = ticker.upper()
    info = _info(ticker)
    if not info:
        return []

    metrics = FinancialMetrics(
        ticker=ticker,
        report_period=end_date,
        period=period,
        currency=info.get("financialCurrency") or "USD",
        market_cap=_num(info.get("marketCap")),
        enterprise_value=_num(info.get("enterpriseValue")),
        price_to_earnings_ratio=_num(info.get("trailingPE")),
        price_to_book_ratio=_num(info.get("priceToBook")),
        price_to_sales_ratio=_num(info.get("priceToSalesTrailing12Months")),
        enterprise_value_to_ebitda_ratio=_num(info.get("enterpriseToEbitda")),
        enterprise_value_to_revenue_ratio=_num(info.get("enterpriseToRevenue")),
        free_cash_flow_yield=None,
        peg_ratio=_num(info.get("pegRatio")) or _num(info.get("trailingPegRatio")),
        gross_margin=_num(info.get("grossMargins")),
        operating_margin=_num(info.get("operatingMargins")),
        net_margin=_num(info.get("profitMargins")),
        return_on_equity=_num(info.get("returnOnEquity")),
        return_on_assets=_num(info.get("returnOnAssets")),
        return_on_invested_capital=None,
        asset_turnover=None,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=_num(info.get("currentRatio")),
        quick_ratio=_num(info.get("quickRatio")),
        cash_ratio=None,
        operating_cash_flow_ratio=None,
        # Yahoo reports debtToEquity as a percentage (e.g. 38.39 == 0.3839x).
        debt_to_equity=(_num(info.get("debtToEquity")) / 100.0 if _num(info.get("debtToEquity")) is not None else None),
        debt_to_assets=None,
        interest_coverage=None,
        revenue_growth=_num(info.get("revenueGrowth")),
        earnings_growth=_num(info.get("earningsGrowth")),
        book_value_growth=None,
        earnings_per_share_growth=None,
        free_cash_flow_growth=None,
        operating_income_growth=None,
        ebitda_growth=None,
        payout_ratio=_num(info.get("payoutRatio")),
        earnings_per_share=_num(info.get("trailingEps")),
        book_value_per_share=_num(info.get("bookValue")),
        free_cash_flow_per_share=None,
    )
    return [metrics]


# ---------------------------------------------------------------------------
# Line items (from annual statements)
# ---------------------------------------------------------------------------
# requested name -> (statement index 0=income/1=balance/2=cashflow, candidate Yahoo row labels)
_DIRECT: dict[str, tuple[int, list[str]]] = {
    "revenue": (0, ["Total Revenue", "Operating Revenue"]),
    "net_income": (0, ["Net Income", "Net Income Common Stockholders"]),
    "operating_income": (0, ["Operating Income", "Total Operating Income As Reported"]),
    "gross_profit": (0, ["Gross Profit"]),
    "cost_of_revenue": (0, ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
    "ebit": (0, ["EBIT"]),
    "ebitda": (0, ["EBITDA", "Normalized EBITDA"]),
    "interest_expense": (0, ["Interest Expense", "Interest Expense Non Operating"]),
    "operating_expense": (0, ["Operating Expense"]),
    "research_and_development": (0, ["Research And Development"]),
    "earnings_per_share": (0, ["Diluted EPS", "Basic EPS"]),
    "total_assets": (1, ["Total Assets"]),
    "total_liabilities": (1, ["Total Liabilities Net Minority Interest"]),
    "shareholders_equity": (1, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    "current_assets": (1, ["Current Assets"]),
    "current_liabilities": (1, ["Current Liabilities"]),
    "cash_and_equivalents": (1, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    "total_debt": (1, ["Total Debt"]),
    "working_capital": (1, ["Working Capital"]),
    "outstanding_shares": (1, ["Ordinary Shares Number", "Share Issued"]),
    "goodwill_and_intangible_assets": (1, ["Goodwill And Other Intangible Assets"]),
    "intangible_assets": (1, ["Other Intangible Assets"]),
    "inventory": (1, ["Inventory"]),
    "free_cash_flow": (2, ["Free Cash Flow"]),
    "capital_expenditure": (2, ["Capital Expenditure"]),
    "depreciation_and_amortization": (2, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Reconciled Depreciation"]),
    "dividends_and_other_cash_distributions": (2, ["Cash Dividends Paid"]),
    "issuance_or_purchase_of_equity_shares": (2, ["Repurchase Of Capital Stock", "Net Common Stock Issuance", "Common Stock Issuance"]),
    "operating_cash_flow": (2, ["Operating Cash Flow"]),
}

# Items derived from other lines when not directly present.
_COMPUTED = {"operating_margin", "gross_margin", "debt_to_equity", "book_value_per_share", "free_cash_flow"}


def _column_for(df, target):
    """Return the statement column whose period end matches `target` (by date), else None."""
    if df is None:
        return None
    for col in df.columns:
        if col == target:
            return col
    return None


def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10, api_key: str = None) -> list[LineItem]:
    ticker = ticker.upper()
    income, balance, cashflow = _stmts(ticker)
    stmts = (income, balance, cashflow)
    if income is None and balance is None and cashflow is None:
        return []

    currency = _info(ticker).get("financialCurrency") or "USD"
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Build the period list from whichever statement has the most columns,
    # keeping only periods on/before end_date, most-recent first.
    base = max((s for s in stmts if s is not None), key=lambda d: len(d.columns), default=None)
    if base is None:
        return []
    periods = sorted([c for c in base.columns if c.to_pydatetime().replace(tzinfo=None) <= end_dt], reverse=True)[:limit]

    results: list[LineItem] = []
    for col in periods:
        cols = [_column_for(s, col) for s in stmts]

        def direct(name):
            idx, labels = _DIRECT[name]
            return _cell(stmts[idx], cols[idx], labels) if cols[idx] is not None else None

        values: dict[str, float | None] = {}
        for name in line_items:
            if name in _DIRECT:
                values[name] = direct(name)
            else:
                values[name] = None  # computed below or unknown -> None

        # Derived items.
        revenue = values.get("revenue") if "revenue" in values else direct("revenue")
        if "operating_margin" in line_items:
            op = values.get("operating_income") or direct("operating_income")
            values["operating_margin"] = (op / revenue) if (op is not None and revenue) else None
        if "gross_margin" in line_items:
            gp = values.get("gross_profit") or direct("gross_profit")
            values["gross_margin"] = (gp / revenue) if (gp is not None and revenue) else None
        if "debt_to_equity" in line_items:
            debt = values.get("total_debt") or direct("total_debt")
            eq = values.get("shareholders_equity") or direct("shareholders_equity")
            values["debt_to_equity"] = (debt / eq) if (debt is not None and eq) else None
        if "book_value_per_share" in line_items:
            eq = values.get("shareholders_equity") or direct("shareholders_equity")
            sh = values.get("outstanding_shares") or direct("outstanding_shares")
            values["book_value_per_share"] = (eq / sh) if (eq is not None and sh) else None
        if line_items.count("free_cash_flow") and values.get("free_cash_flow") is None:
            ocf = direct("operating_cash_flow")
            capex = direct("capital_expenditure")
            if ocf is not None and capex is not None:
                values["free_cash_flow"] = ocf + capex  # capex is negative in the cashflow stmt

        results.append(
            LineItem(
                ticker=ticker,
                report_period=col.strftime("%Y-%m-%d"),
                period=period,
                currency=currency,
                **values,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Company news (no sentiment available from Yahoo)
# ---------------------------------------------------------------------------
def get_company_news(ticker: str, end_date: str, start_date: str = None, limit: int = 1000, api_key: str = None) -> list[CompanyNews]:
    ticker = ticker.upper()
    if ticker not in _NEWS_CACHE:
        try:
            _NEWS_CACHE[ticker] = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            _warn(f"news({ticker}) failed: {e}")
            _NEWS_CACHE[ticker] = []

    out: list[CompanyNews] = []
    for item in _NEWS_CACHE[ticker]:
        content = item.get("content", item)  # newer yfinance nests under 'content'
        title = content.get("title") or ""
        provider = content.get("provider") or {}
        publisher = provider.get("displayName") if isinstance(provider, dict) else (item.get("publisher") or "")
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            ref = content.get(key)
            if isinstance(ref, dict) and ref.get("url"):
                url = ref["url"]
                break
        url = url or item.get("link") or ""

        date = content.get("pubDate") or content.get("displayTime") or ""
        if not date and item.get("providerPublishTime"):
            date = datetime.utcfromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")
        date = (date or "")[:10]

        if not title:
            continue
        if start_date and date and date < start_date:
            continue
        if end_date and date and date > end_date:
            continue
        out.append(CompanyNews(ticker=ticker, title=title, author=publisher or "", source=publisher or "", date=date, url=url, sentiment=None))

    return out[:limit]


# ---------------------------------------------------------------------------
# Insider trades (partial fidelity)
# ---------------------------------------------------------------------------
def get_insider_trades(ticker: str, end_date: str, start_date: str = None, limit: int = 1000, api_key: str = None) -> list[InsiderTrade]:
    ticker = ticker.upper()
    if ticker not in _INSIDER_CACHE:
        try:
            _INSIDER_CACHE[ticker] = yf.Ticker(ticker).insider_transactions
        except Exception as e:  # noqa: BLE001
            _warn(f"insider_transactions({ticker}) failed: {e}")
            _INSIDER_CACHE[ticker] = None

    df = _INSIDER_CACHE[ticker]
    if df is None or getattr(df, "empty", True):
        return []

    issuer = _info(ticker).get("longName")

    def pick(row, *names):
        for n in names:
            if n in row and row[n] is not None and not (isinstance(row[n], float) and math.isnan(row[n])):
                return row[n]
        return None

    out: list[InsiderTrade] = []
    for _, row in df.iterrows():
        row = row.to_dict()
        raw_date = pick(row, "Start Date", "Date")
        date = str(raw_date)[:10] if raw_date is not None else None
        if start_date and date and date < start_date:
            continue
        if end_date and date and date > end_date:
            continue
        out.append(
            InsiderTrade(
                ticker=ticker,
                issuer=issuer,
                name=pick(row, "Insider"),
                title=pick(row, "Position"),
                is_board_director=None,
                transaction_date=date,
                transaction_shares=_num(pick(row, "Shares")),
                transaction_price_per_share=None,
                transaction_value=_num(pick(row, "Value")),
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=pick(row, "Transaction", "Text"),
                filing_date=date or end_date,
            )
        )
    return out[:limit]
