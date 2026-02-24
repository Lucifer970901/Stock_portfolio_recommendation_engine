import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# ── Shared test data ───────────────────────────────────────────────────────────

TICKERS = ['AAPL', 'MSFT', 'JNJ', 'XOM']

@pytest.fixture
def sample_prices():
    """Fake price history for 4 stocks over 300 days"""
    np.random.seed(42)
    dates   = pd.date_range('2023-01-01', periods=300, freq='B')
    data    = {}
    for ticker in TICKERS:
        start = np.random.uniform(100, 500)
        returns = np.random.normal(0.0005, 0.02, 300)
        prices  = start * np.cumprod(1 + returns)
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_fundamentals():
    """Fake fundamentals for 4 stocks"""
    return pd.DataFrame({
        'sector':         ['Technology', 'Technology', 'Healthcare', 'Energy'],
        'pe_ratio':       [28.0, 32.0, 18.0, 12.0],
        'pb_ratio':       [3.5,  4.0,  2.0,  1.5],
        'roe':            [0.25, 0.30, 0.15, 0.10],
        'debt_to_equity': [1.2,  0.8,  0.5,  0.9],
        'revenue_growth': [0.08, 0.12, 0.05, 0.03],
        'dividend_yield': [0.01, 0.01, 0.03, 0.04],
        'beta':           [1.2,  1.1,  0.7,  0.9],
        'market_cap':     [2e12, 2.5e12, 4e11, 3e11],
    }, index=TICKERS)

@pytest.fixture
def sample_combined(sample_fundamentals, sample_prices):
    """Merged fundamentals + technical features"""
    from app.features.technical import compute_technical_features
    from app.features.fundamental import merge_features
    technical = compute_technical_features(sample_prices)
    return merge_features(sample_fundamentals, technical)

@pytest.fixture
def sample_scaled(sample_combined):
    """Scaled feature matrix"""
    from app.features.fundamental import scale_features
    scaled, _, _ = scale_features(sample_combined)
    return scaled