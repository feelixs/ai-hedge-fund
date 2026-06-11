# Dip scanner — flags sharp stock-specific drops in watchlist.txt, then blocks
# while Claude Code judges each one (run /judge-dips in a session in this repo).
#
# WARNING: do not START this while ./run.sh (main.py) is starting up — both wipe
# claude_agent/prompts/ on launch and would clobber each other's in-flight files.
# Coexistence after startup is fine (filename prefixes are disjoint).
export DATA_SOURCE="${DATA_SOURCE:-yfinance}"
poetry run python -m src.dip.scanner "$@"
