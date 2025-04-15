#!/bin/zsh

# Get the ticker list from Python config
ticker_list=$(python3 -c "from config import tickers; print(','.join(tickers))")

# Run the poetry command with the formatted tickers
poetry run python src/main.py --ticker "$ticker_list"
