import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.data.fetcher import fetch_prices, fetch_fundamentals

TICKERS = ["AAPL", "MSFT"]


def make_fake_prices():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    data  = {t: np.random.uniform(100, 500, 100) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


def make_fake_ticker_mock():
    """
    Build a mock yf.Ticker that satisfies the new fetcher:
    - .history()               → price DataFrame
    - .quarterly_financials    → revenue DataFrame
    - .quarterly_balance_sheet → balance sheet DataFrame
    - .quarterly_earnings      → earnings DataFrame (EPS)
    - .info                    → dict (pb, roe, beta, etc.)
    """
    mock = MagicMock()

    # ── Price history ─────────────────────────────────────────────────────────
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    mock.history.return_value = pd.DataFrame(
        {"Close": [150.0, 151.0, 152.0, 153.0, 154.0]}, index=dates
    )

    # ── Quarterly financials (revenue) ────────────────────────────────────────
    q_dates = pd.date_range("2023-01-01", periods=8, freq="QE")
    mock.quarterly_financials = pd.DataFrame(
        {d: [1e10, 2e9] for d in q_dates},
        index=["Total Revenue", "Operating Income"],
    )

    # ── Quarterly balance sheet ───────────────────────────────────────────────
    mock.quarterly_balance_sheet = pd.DataFrame(
        {d: [5e9, 2e9] for d in q_dates},
        index=["Total Debt", "Stockholders Equity"],
    )

    # ── Quarterly earnings (EPS) ──────────────────────────────────────────────
    mock.quarterly_earnings = pd.DataFrame(
        {"Earnings": [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]},
        index=pd.date_range("2022-01-01", periods=8, freq="QE"),
    )

    # ── .info (supplemental fields) ───────────────────────────────────────────
    mock.info = {
        "sector":         "Technology",
        "priceToBook":    3.5,
        "returnOnEquity": 0.25,
        "debtToEquity":   1.2,
        "revenueGrowth":  0.08,
        "dividendYield":  0.01,
        "beta":           1.2,
        "marketCap":      2e12,
    }

    return mock


# ─────────────────────────────────────────────────────────────────────────────
# fetch_prices tests (unchanged behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_prices_returns_dataframe():
    """fetch_prices should return a DataFrame with ticker columns."""
    with patch("yfinance.download") as mock_dl:
        fake = make_fake_prices()
        mock_dl.return_value = {"Close": fake}
        result = fetch_prices(TICKERS, period="1y")
        assert isinstance(result, pd.DataFrame)
        mock_dl.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# fetch_fundamentals tests (updated for parallel + PIT + disk cache)
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_fundamentals_returns_dataframe():
    """fetch_fundamentals should return DataFrame indexed by ticker."""
    with patch("yfinance.Ticker", return_value=make_fake_ticker_mock()), \
         patch("app.data.fetcher._disk_cache") as mock_cache:

        mock_cache.get.return_value = None  # force cache miss → API call

        result = fetch_fundamentals(TICKERS)

        assert isinstance(result, pd.DataFrame)
        assert "sector"         in result.columns
        assert "pe_ratio"       in result.columns
        assert "eps_ttm"        in result.columns   # new PIT column
        assert "as_of_date"     in result.columns   # new PIT column
        assert "revenue_growth" in result.columns
        assert "debt_to_equity" in result.columns
        assert set(TICKERS).issubset(set(result.index.tolist()))


def test_fetch_fundamentals_handles_missing_fields():
    """fetch_fundamentals should handle missing fields with NaN."""
    sparse_mock = MagicMock()
    sparse_mock.history.return_value    = pd.DataFrame()
    sparse_mock.quarterly_financials    = pd.DataFrame()
    sparse_mock.quarterly_balance_sheet = pd.DataFrame()
    sparse_mock.quarterly_earnings      = pd.DataFrame()
    sparse_mock.info                    = {"sector": "Technology"}

    with patch("yfinance.Ticker", return_value=sparse_mock), \
         patch("app.data.fetcher._disk_cache") as mock_cache:

        mock_cache.get.return_value = None

        result = fetch_fundamentals(["AAPL"])

        assert isinstance(result, pd.DataFrame)
        if "AAPL" in result.index:
            assert pd.isna(result.loc["AAPL", "pe_ratio"])
            assert pd.isna(result.loc["AAPL", "eps_ttm"])


def test_fetch_fundamentals_handles_errors():
    """fetch_fundamentals should skip tickers that throw errors after retries."""
    with patch("yfinance.Ticker", side_effect=Exception("API error")), \
         patch("app.data.fetcher._disk_cache") as mock_cache, \
         patch("app.data.fetcher._make_retry", return_value=lambda f: f):
        # disable tenacity retry to keep test fast
        mock_cache.get.return_value = None

        result = fetch_fundamentals(TICKERS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def test_fetch_fundamentals_uses_cache():
    """fetch_fundamentals should return cached data without calling yfinance."""
    cached_data = {
        "ticker":         "AAPL",
        "pe_ratio":       28.0,
        "eps_ttm":        6.0,
        "as_of_date":     "2024-01-01",
        "revenue_growth": 0.08,
        "debt_to_equity": 1.2,
        "sector":         "Technology",
        "pb_ratio":       3.5,
        "roe":            0.25,
        "dividend_yield": 0.01,
        "beta":           1.2,
        "market_cap":     2e12,
    }

    with patch("yfinance.Ticker") as mock_ticker, \
         patch("app.data.fetcher._disk_cache") as mock_cache:

        mock_cache.get.return_value = cached_data  # cache hit

        result = fetch_fundamentals(["AAPL"])

        # yfinance should never be called
        mock_ticker.assert_not_called()
        assert isinstance(result, pd.DataFrame)
        assert result.loc["AAPL", "pe_ratio"] == 28.0


def test_fetch_fundamentals_pit_cutoff_date():
    """fetch_fundamentals should pass cutoff_date through to PIT calculations."""
    cutoff = datetime(2023, 6, 30)

    with patch("yfinance.Ticker", return_value=make_fake_ticker_mock()), \
         patch("app.data.fetcher._disk_cache") as mock_cache:

        mock_cache.get.return_value = None

        result = fetch_fundamentals(["AAPL"], cutoff_date=cutoff)

        if "AAPL" in result.index:
            assert "2023" in str(result.loc["AAPL", "as_of_date"])