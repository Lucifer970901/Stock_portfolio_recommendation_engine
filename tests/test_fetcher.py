import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.data.fetcher import fetch_prices, fetch_fundamentals

TICKERS = ['AAPL', 'MSFT']

def make_fake_prices():
    dates  = pd.date_range('2023-01-01', periods=100, freq='B')
    data   = {t: np.random.uniform(100, 500, 100) for t in TICKERS}
    return pd.DataFrame(data, index=dates)

def test_fetch_prices_returns_dataframe():
    """fetch_prices should return a DataFrame with ticker columns"""
    with patch('yfinance.download') as mock_dl:
        fake = make_fake_prices()
        mock_dl.return_value = {'Close': fake}
        result = fetch_prices(TICKERS, period='1y')
        assert isinstance(result, pd.DataFrame)
        mock_dl.assert_called_once()

def test_fetch_fundamentals_returns_dataframe():
    """fetch_fundamentals should return DataFrame indexed by ticker"""
    mock_info = {
        'sector': 'Technology',
        'trailingPE': 28.0,
        'priceToBook': 3.5,
        'returnOnEquity': 0.25,
        'debtToEquity': 1.2,
        'revenueGrowth': 0.08,
        'dividendYield': 0.01,
        'beta': 1.2,
        'marketCap': 2e12,
    }
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.info = mock_info
        result = fetch_fundamentals(TICKERS)
        assert isinstance(result, pd.DataFrame)
        assert 'sector' in result.columns
        assert 'pe_ratio' in result.columns
        assert result.index.tolist() == TICKERS

def test_fetch_fundamentals_handles_missing_fields():
    """fetch_fundamentals should handle missing fields with NaN"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.info = {'sector': 'Technology'}
        result = fetch_fundamentals(['AAPL'])
        assert result.loc['AAPL', 'pe_ratio'] != result.loc['AAPL', 'pe_ratio']  # NaN check

def test_fetch_fundamentals_handles_errors():
    """fetch_fundamentals should skip tickers that throw errors"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.side_effect = Exception("API error")
        result = fetch_fundamentals(['AAPL', 'MSFT'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0