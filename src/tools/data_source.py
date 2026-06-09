"""Selects which market-data backend ``src/tools/api.py`` routes to.

Controlled by the ``DATA_SOURCE`` environment variable (set directly or via the
``--data-source`` CLI flag). Defaults to ``financialdatasets``.
"""

import os

FINANCIAL_DATASETS = "financialdatasets"
YFINANCE = "yfinance"


def get_data_source() -> str:
    """Return the active data source, lowercased. Defaults to financialdatasets."""
    return (os.environ.get("DATA_SOURCE") or FINANCIAL_DATASETS).strip().lower()
