#!/bin/bash
# Run the stock scanner with default tickers

poetry run python src/scanner.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,AMD,INTC "$@"
