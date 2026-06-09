# yfinance as an alternative data source — scoping & design

## Problem

The app's data layer (`src/tools/api.py`) is hardwired to
`financialdatasets.ai`, whose free tier only covers AAPL, GOOGL, MSFT, NVDA,
TSLA. Micro-caps like EDBL require a paid `FINANCIAL_DATASETS_API_KEY` (and the
run was also hitting transient TLS errors). We want a free, no-key source that
works for arbitrary tickers.

**Chosen source:** `yfinance` (Yahoo Finance) — free, no key, broad coverage.
**Chosen fidelity for metrics:** current snapshot (~20 of ~40 fields); the rest
left `None`. (See decision log below.)

## Data surface the agents actually use

| Function | Calls | Returns |
|---|---|---|
| `get_financial_metrics` | 29 | series of `FinancialMetrics` (~40 ratio fields) |
| `search_line_items` | 27 | raw statement lines (29 distinct items) |
| `get_market_cap` | 27 | market cap at a date |
| `get_insider_trades` | 16 | `InsiderTrade` rows |
| `get_company_news` | 16 | `CompanyNews` (incl. `sentiment`) |
| `get_prices` | 9 | OHLCV `Price` rows |

Distinct line items requested (by frequency): free_cash_flow, revenue,
net_income, outstanding_shares, operating_margin, capital_expenditure,
total_debt, operating_income, cash_and_equivalents, gross_margin,
shareholders_equity, total_assets, total_liabilities, earnings_per_share, ebit,
dividends_and_other_cash_distributions, depreciation_and_amortization,
current_assets, current_liabilities, issuance_or_purchase_of_equity_shares,
ebitda, research_and_development, gross_profit, debt_to_equity,
interest_expense, operating_expense, intangible_assets, working_capital,
return_on_invested_capital, goodwill_and_intangible_assets,
book_value_per_share.

## Coverage map (yfinance)

| Need | yfinance source | Verdict |
|---|---|---|
| Prices (OHLCV) | `Ticker.history(start,end,interval)` | Full — direct map to `Price` |
| Line items | `income_stmt` / `balance_sheet` / `cashflow` (annual) | Most map via a label table; margins + fcf + working_capital + book_value_per_share computed |
| Market cap | `fast_info["market_cap"]` / `info["marketCap"]` | Current only; historical date approximated as `shares × close(end_date)` |
| News | `Ticker.news` | title/url/publisher/date map; `sentiment` always `None` |
| Insider trades | `insider_transactions` | Partial — shares/value/name/title/date; no before/after counts, no board flag |
| Financial metrics | `.info` (snapshot) | ~20 fields mapped; ~20 left `None` (per decision) |

### Snapshot metrics mapped from `.info`
market_cap, enterprise_value, price_to_earnings_ratio, price_to_book_ratio,
price_to_sales_ratio, enterprise_value_to_ebitda_ratio,
enterprise_value_to_revenue_ratio, peg_ratio, gross_margin, operating_margin,
net_margin, return_on_equity, return_on_assets, current_ratio, quick_ratio,
debt_to_equity, revenue_growth, earnings_growth, payout_ratio,
earnings_per_share, book_value_per_share. Remainder (`return_on_invested_capital`,
turnovers, `days_sales_outstanding`, `operating_cycle`, `cash_ratio`,
`interest_coverage`, `debt_to_assets`, per-period growth fields, FCF
yield/per-share) → `None`.

## Known limitations (accepted for v1)

1. **Single-period metrics.** `.info` is one current snapshot, not a per-period
   series; `limit` is effectively 1. Trend/consistency agents see one point.
2. **Shallow history.** yfinance gives ~4 annual periods; `period="ttm"` is
   approximated with annual statements. No deep TTM series.
3. **No news sentiment.** Rule-based Sentiment agent's news arm degrades to
   neutral; LLM News Sentiment agent still reads titles.
4. **Lower insider fidelity.**
5. **yfinance is unofficial** — rate limits (HTTP 429) and occasional schema
   drift; fine for a few tickers, not heavy backtests.

All return types are unchanged and every model field is `Optional`, so agents
run; only signal quality on fundamentals-heavy agents drops.

## Implementation

### `src/tools/data_source.py` (new)
`get_data_source() -> str` reads `os.environ["DATA_SOURCE"]` (default
`"financialdatasets"`), lowercased. One small helper, imported where needed.

### `src/tools/yfinance_provider.py` (new)
Implements `get_prices`, `get_financial_metrics`, `search_line_items`,
`get_market_cap`, `get_company_news`, `get_insider_trades` with signatures
matching `api.py`, returning the same model objects.

- `import yfinance as yf` at top — hard dependency, **no silent fallback** if the
  import fails (per project rule).
- **Per-ticker memoization** of raw fetches (`info`, statements, `news`,
  `insider_transactions`) in module-level dicts, so the 29 metric calls / 27
  line-item calls don't re-hit Yahoo and trip rate limits.
- A label-lookup helper tries candidate Yahoo row names per requested item;
  computed items (fcf, margins, working_capital, book_value_per_share) derived.
- On a genuine data-fetch failure, return an empty list / `None` (matching the
  existing `api.py` contract that agents already tolerate) and print a warning
  to stderr — never crash the run.

### `src/tools/api.py` (dispatch)
A small `@_route` decorator on each of the 6 public functions: if
`get_data_source() == "yfinance"`, forward `*args, **kwargs` to the matching
`yfinance_provider` function; otherwise run the existing financialdatasets
implementation unchanged. Agents keep importing from `src.tools.api`.

### `src/cli/input.py` (toggle)
Add `--data-source {financialdatasets,yfinance}` (default `financialdatasets`)
in `add_common_args`; `parse_cli_inputs` sets `os.environ["DATA_SOURCE"]` from
it so the dispatch picks it up. Also honored via the env var directly.

### `pyproject.toml`
Add `yfinance` dependency.

## Decision log
- **Data source:** yfinance (free, no key, covers EDBL).
- **Metrics fidelity:** current snapshot (~20 fields), rest `None`. Computed
  per-period metrics deferred — revisit if signal quality is insufficient.

## Out of scope (v1)
- Per-period computed metrics series, deep TTM, news sentiment scoring,
  full insider-trade fidelity, keyed providers (FMP/Alpha Vantage).
