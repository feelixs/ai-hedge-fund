if [ -n "$1" ]; then
    poetry run python -m src.main --tickers "$1"
else
    poetry run python -m src.main
fi
