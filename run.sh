# Default to the free, no-key yfinance data source. Override by exporting
# DATA_SOURCE=financialdatasets or passing --data-source financialdatasets.
export DATA_SOURCE="${DATA_SOURCE:-yfinance}"

if [ -n "$1" ]; then
    poetry run python -m src.main --tickers "$1" "${@:2}"
else
    poetry run python -m src.main
fi
status=$?
if [ $status -eq 0 ]; then
    echo
    echo "Tip: run /dispatch-ta in Claude Code for end-of-week TA price targets."
fi
exit $status
