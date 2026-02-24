import pytest
import pandas as pd
import numpy as np
from app.features.technical import compute_rsi, compute_technical_features
from app.features.fundamental import merge_features, scale_features, FEATURE_COLS

def test_rsi_returns_float(sample_prices):
    """RSI should return a float between 0 and 100"""
    close  = sample_prices['AAPL']
    result = compute_rsi(close)
    assert isinstance(float(result), float)
    assert 0 <= result <= 100

def test_rsi_short_series():
    """RSI should handle short series gracefully"""
    close  = pd.Series([100, 101, 102, 103, 104])
    result = compute_rsi(close, window=3)
    assert result is not None

def test_technical_features_shape(sample_prices):
    """Technical features should have correct shape"""
    result = compute_technical_features(sample_prices)
    assert isinstance(result, pd.DataFrame)
    assert 'momentum_3m' in result.columns
    assert 'momentum_6m' in result.columns
    assert 'volatility'  in result.columns
    assert 'rsi'         in result.columns
    assert len(result) == len(sample_prices.columns)

def test_technical_features_skips_short_series():
    """Stocks with less than 60 days should be skipped"""
    dates  = pd.date_range('2023-01-01', periods=30, freq='B')
    prices = pd.DataFrame({'AAPL': np.random.uniform(100, 200, 30)}, index=dates)
    result = compute_technical_features(prices)
    assert len(result) == 0

def test_merge_features(sample_fundamentals, sample_prices):
    """merge_features should join fundamentals and technical correctly"""
    from app.features.technical import compute_technical_features
    technical = compute_technical_features(sample_prices)
    result    = merge_features(sample_fundamentals, technical)
    assert isinstance(result, pd.DataFrame)
    assert 'pe_ratio'    in result.columns
    assert 'momentum_6m' in result.columns

def test_scale_features_shape(sample_combined):
    """scale_features should return same shape as input"""
    scaled, scaler, imputer = scale_features(sample_combined)
    assert scaled.shape == (len(sample_combined), len(FEATURE_COLS))

def test_scale_features_no_nulls(sample_combined):
    """scaled features should have no missing values"""
    scaled, _, _ = scale_features(sample_combined)
    assert scaled.isnull().sum().sum() == 0

def test_scale_features_standardized(sample_combined):
    """scaled features should have mean ~0 and std ~1"""
    scaled, _, _ = scale_features(sample_combined)
    assert abs(scaled.mean().mean()) < 0.1
    assert abs(scaled.std().mean() - 1.0) < 0.3