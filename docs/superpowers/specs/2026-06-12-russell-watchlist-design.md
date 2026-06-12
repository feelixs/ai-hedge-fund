# Russell 1000 Watchlist Builder — Design

**Date:** 2026-06-12
**Status:** Approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-06-11-dip-scanner-design.md`

## Purpose

Replace the hand-seeded `watchlist.txt` (48 one-shotted mega-caps) with the full
Russell 1000, sourced from the iShares IWB ETF's daily holdings CSV. The dip
scanner then watches ~1,000 names instead of ~50. Because a 1,000-ticker
sequential scan would take ~15 minutes, the scanner's price fetching is
parallelized in the same change.

The user accepted the trade-off that the watchlist is no longer hand-curated:
the math packet and the `/judge-dips` research step carry the quality
assessment per dip, and the 10-candidate cap keeps AI usage flat regardless of
universe size.

## Revision (2026-06-12, same day)

The iShares endpoint serves a TLS-fingerprinting bot wall to all scripted
clients (HTTP 200 + `text/csv` content-type, HTML body) — confirmed with full
browser headers via both requests and curl. **Source switched to the Wikipedia
"Russell 1000 Index" components table** (user's choice over a manual-download
`--input` flag): 1,004 rows with Company/Symbol/GICS Sector columns, verified
script-accessible and current to the April 2026 reconstitution. Parser becomes
`parse_wikipedia_constituents(html)` via `pandas.read_html` (new dep: lxml);
the table is selected by its column names (has both `Symbol` and `Company`),
never by position or size. Known trade-off: community-maintained, may lag a
future reconstitution by weeks. Everything else (renderer, atomic writer,
MIN_TICKERS guard, CLI) is unchanged.

## Key decisions

| Decision | Choice |
|----------|--------|
| Source | ~~iShares IWB holdings CSV~~ → Wikipedia Russell 1000 components table (see Revision) |
| Integration | Script REPLACES `watchlist.txt` (the default `./dip.sh` universe) |
| Refresh model | Manual re-run a few times a year; no scheduling, no caching |
| Scan latency fix | Parallelize `fetch_price_dfs` (~8 workers) |

## Components

### 1. `scripts/build_watchlist.py` (new)

Two cleanly separated units:

**`parse_ishares_holdings(csv_text: str) -> list[tuple[str, str, str]]`** — pure
function, no network. Returns `(ticker, name, sector)` tuples.

- Finds the real header row dynamically (the line starting with `Ticker,`),
  skipping iShares' ~9 metadata lines and any footer disclaimer rows.
- Keeps only rows with `Asset Class == "Equity"`.
- Drops cash/derivative placeholders (blank tickers, tickers containing `_`,
  e.g. `XTSLA`, `MARGIN_USD`).
- Normalizes tickers for yfinance: uppercase; `.`, `/`, and spaces → `-`
  (`BRK.B` → `BRK-B`).
- Deduplicates, preserving CSV order.

**`main()`** — CLI glue:

- Downloads the IWB holdings CSV via `requests` (existing dependency) from the
  iShares product-page CSV endpoint.
- Parses with `parse_ishares_holdings`.
- **Sanity guard:** fewer than 500 tickers → abort loudly (malformed download
  must not produce a quietly tiny watchlist).
- Writes the output file: a generated-file header comment (source URL, date,
  count, regeneration command), then one ticker per line with
  `# Company Name — Sector` trailing comments. The existing `load_watchlist`
  strips comments, so the file stays human-readable and hand-prunable.
- **Atomic write:** write to `<output>.tmp`, then `os.replace` onto the target
  only on success. A failed download or parse can never clobber the working
  watchlist.
- `--output` flag (default `watchlist.txt` at the project root).

Usage: `poetry run python scripts/build_watchlist.py`

### 2. Parallel `fetch_price_dfs` (modify `src/dip/scanner.py`)

- ThreadPoolExecutor, `MAX_FETCH_WORKERS = 8` module constant (deliberately
  modest — yfinance throttles aggressive parallelism).
- Identical warn-and-skip semantics per ticker (exception → print + skip;
  empty → print + skip). Result dict contents are order-independent for
  `detect_dips`.

## Error handling

- Download failure (non-200, timeout, connection error) → print the error,
  exit non-zero, watchlist untouched.
- Parse yields < 500 tickers → abort loudly, watchlist untouched.
- No retries, no caching: the script runs a few times a year, manually.

## Testing

- Parser unit tests against a canned CSV string fixture covering: metadata
  preamble, equity filter, cash/futures rows, `BRK.B` → `BRK-B`, dedup, footer
  junk.
- Sanity-guard test (< 500 tickers aborts).
- Atomic-write test (failure path leaves an existing output file unchanged).
- Parallel `fetch_price_dfs`: existing monkeypatch test must still pass; add an
  assertion-equivalent test that results match the sequential behavior (same
  tickers kept/skipped).
- Manual: run the script for real, confirm ~1,000 names, run `./dip.sh` and
  confirm scan completes in minutes.

## Out of scope (YAGNI)

- Scheduled/automatic refresh.
- Other ETF sources (QUAL/MOAT/SCHD) — the parser is shaped to extend to them
  later, but only IWB ships now.
- Sector filtering, weight-based ordering, market-cap floors.
- Retry/backoff logic in the downloader.
