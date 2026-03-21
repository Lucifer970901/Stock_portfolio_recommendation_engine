import pytest
import pandas as pd
import numpy as np
from app.features.technical import compute_rsi, compute_technical_features
from app.features.fundamentals import (
    merge_features,
    scale_features,
    FEATURE_COLS,
    ENGINEERED_COLS,
)

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """Realistic price DataFrame with 200 days for 5 tickers."""
    dates = pd.date_range('2023-01-01', periods=200, freq='B')
    data  = {t: np.cumprod(1 + np.random.normal(0.0005, 0.015, 200)) * 100
             for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_fundamentals():
    """Realistic fundamentals DataFrame covering edge cases."""
    return pd.DataFrame({
        'pe_ratio':       [28.0,  35.0,  25.0,  np.nan, 21.0],   # JPM has NaN
        'pb_ratio':       [3.5,   12.0,  6.0,   1.2,    3.8],
        'roe':            [0.25,  0.40,  0.22,  0.12,   0.18],
        'debt_to_equity': [1.2,   0.5,   0.3,   1.8,    0.6],
        'revenue_growth': [0.06,  0.15,  0.14,  0.09,   0.04],
        'dividend_yield': [0.005, 0.0,   0.0,   0.025,  0.03],   # MSFT/GOOGL no div
        'beta':           [1.2,   0.9,   1.1,   1.3,    0.7],
        'market_cap':     [3e12,  2.8e12, 2e12, 5e11,   4e11],
        'eps_ttm':        [8.0,   16.0,  22.0,  4.2,    11.0],
        'as_of_date':     ['2024-01-01'] * 5,
        'sector':         ['Technology', 'Technology', 'Communication Services',
                           'Financial Services', 'Healthcare'],
    }, index=TICKERS)


@pytest.fixture
def sample_fundamentals_with_edge_cases():
    """Fundamentals with negative equity, loss-making, and no-dividend tickers."""
    return pd.DataFrame({
        'pe_ratio':       [-5.0,  300.0,  25.0],   # loss-making, extreme PE, normal
        'pb_ratio':       [3.5,   12.0,   6.0],
        'roe':            [-0.5,  0.40,   0.22],    # negative ROE
        'debt_to_equity': [-25.0, 0.5,    0.3],     # negative equity (MCD-like)
        'revenue_growth': [0.06,  1.14,   0.14],    # extreme growth (NVDA-like)
        'dividend_yield': [0.0,   0.0,    0.025],
        'beta':           [1.2,   0.9,    1.1],
        'market_cap':     [3e12,  2.8e12, 2e12],
        'eps_ttm':        [-1.0,  5.0,    22.0],
        'as_of_date':     ['2024-01-01'] * 3,
        'sector':         ['Technology', 'Technology', 'Healthcare'],
    }, index=['INTC', 'NVDA', 'GOOGL'])


@pytest.fixture
def sample_technical(sample_prices):
    return compute_technical_features(sample_prices)


@pytest.fixture
def sample_combined(sample_fundamentals, sample_technical):
    return merge_features(sample_fundamentals, sample_technical)


# ── RSI tests ─────────────────────────────────────────────────────────────────

def test_rsi_returns_float(sample_prices):
    """RSI should return a float between 0 and 100."""
    close  = sample_prices['AAPL']
    result = compute_rsi(close)
    assert isinstance(float(result), float)
    assert 0 <= result <= 100


def test_rsi_short_series():
    """RSI should handle short series gracefully."""
    close  = pd.Series([100, 101, 102, 103, 104])
    result = compute_rsi(close, window=3)
    assert result is not None


# ── Technical feature tests ───────────────────────────────────────────────────

def test_technical_features_shape(sample_prices):
    """Technical features should have correct columns and rows."""
    result = compute_technical_features(sample_prices)
    assert isinstance(result, pd.DataFrame)
    assert 'momentum_3m' in result.columns
    assert 'momentum_6m' in result.columns
    assert 'volatility'  in result.columns
    assert 'rsi'         in result.columns
    assert len(result) == len(sample_prices.columns)


def test_technical_features_skips_short_series():
    """Stocks with less than 60 days of history should be skipped."""
    dates  = pd.date_range('2023-01-01', periods=30, freq='B')
    prices = pd.DataFrame({'AAPL': np.random.uniform(100, 200, 30)}, index=dates)
    result = compute_technical_features(prices)
    assert len(result) == 0


def test_volatility_is_annualized(sample_prices):
    """Volatility should be annualized (roughly 0.1 to 0.6 for normal stocks)."""
    result = compute_technical_features(sample_prices)
    assert result['volatility'].between(0.05, 1.0).all()


# ── merge_features tests ──────────────────────────────────────────────────────

def test_merge_features_shape(sample_fundamentals, sample_technical):
    """merge_features should produce inner join on ticker index."""
    result = merge_features(sample_fundamentals, sample_technical)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_fundamentals)
    assert len(result) <= len(sample_technical)


def test_merge_features_has_both_cols(sample_fundamentals, sample_technical):
    """Merged DataFrame should contain both fundamental and technical columns."""
    result = merge_features(sample_fundamentals, sample_technical)
    assert 'pe_ratio'    in result.columns
    assert 'momentum_3m' in result.columns
    assert 'momentum_6m' in result.columns
    assert 'volatility'  in result.columns


# ── scale_features shape and null tests ──────────────────────────────────────

def test_scale_features_shape(sample_combined):
    """scale_features should return DataFrame with correct number of columns."""
    scaled, scaler, imputer = scale_features(sample_combined)
    assert isinstance(scaled, pd.DataFrame)
    available_cols = [c for c in FEATURE_COLS if c in scaled.columns]
    assert scaled.shape[1] == len(available_cols)


def test_scale_features_no_nulls(sample_combined):
    """Scaled features should have no missing values."""
    scaled, _, _ = scale_features(sample_combined)
    assert scaled.isnull().sum().sum() == 0


def test_scale_features_standardized(sample_combined):
    """Scaled features should have mean ~0 and std ~1."""
    scaled, _, _ = scale_features(sample_combined)
    assert abs(scaled.mean().mean()) < 0.1
    assert abs(scaled.std().mean() - 1.0) < 0.3


# ── Engineered feature tests ──────────────────────────────────────────────────

def test_is_profitable_flag(sample_combined):
    """is_profitable should be 1 for positive PE, 0 for NaN or negative PE."""
    scaled, _, _ = scale_features(sample_combined)
    # Check that is_profitable exists in output
    assert 'is_profitable' in scaled.columns


def test_is_profitable_correct_values(sample_fundamentals, sample_technical):
    """is_profitable should correctly identify profitable vs non-profitable."""
    from app.features.fundamentals import _engineer_features
    merged      = merge_features(sample_fundamentals, sample_technical)
    engineered  = _engineer_features(merged)
    # JPM has NaN pe_ratio -> should be 0
    assert engineered.loc['JPM', 'is_profitable'] == 0
    # AAPL has pe_ratio=28 -> should be 1
    assert engineered.loc['AAPL', 'is_profitable'] == 1


def test_dividend_split(sample_fundamentals, sample_technical):
    """dividend_yield should be split into has_dividend and dividend_yield_amount."""
    from app.features.fundamentals import _engineer_features
    merged     = merge_features(sample_fundamentals, sample_technical)
    engineered = _engineer_features(merged)

    # MSFT and GOOGL have dividend_yield=0 -> has_dividend=0
    assert engineered.loc['MSFT',  'has_dividend'] == 0
    assert engineered.loc['GOOGL', 'has_dividend'] == 0

    # AAPL has dividend_yield=0.005 -> has_dividend=1
    assert engineered.loc['AAPL', 'has_dividend'] == 1

    # dividend_yield_amount should match original yield
    assert engineered.loc['AAPL', 'dividend_yield_amount'] == pytest.approx(0.005)
    assert engineered.loc['MSFT', 'dividend_yield_amount'] == 0.0


def test_market_cap_log_transform(sample_fundamentals, sample_technical):
    """market_cap_log should be log1p of market_cap."""
    from app.features.fundamentals import _engineer_features
    merged     = merge_features(sample_fundamentals, sample_technical)
    engineered = _engineer_features(merged)

    expected = np.log1p(sample_fundamentals.loc['AAPL', 'market_cap'])
    assert engineered.loc['AAPL', 'market_cap_log'] == pytest.approx(expected)


def test_has_momentum_history(sample_combined):
    """has_momentum_history should be 1 when both momentum cols are present."""
    from app.features.fundamentals import _engineer_features
    engineered = _engineer_features(sample_combined)
    # All tickers in sample_combined have 200 days of prices -> momentum available
    assert engineered['has_momentum_history'].all()


def test_has_momentum_history_missing():
    """has_momentum_history should be 0 when momentum cols are NaN."""
    from app.features.fundamentals import _engineer_features
    df = pd.DataFrame({
        'pe_ratio':       [28.0],
        'pb_ratio':       [3.5],
        'roe':            [0.25],
        'debt_to_equity': [1.2],
        'revenue_growth': [0.06],
        'dividend_yield': [0.005],
        'beta':           [1.2],
        'market_cap':     [3e12],
        'momentum_3m':    [np.nan],   # missing
        'momentum_6m':    [np.nan],   # missing
        'volatility':     [0.2],
        'rsi':            [55.0],
    }, index=['AAPL'])
    engineered = _engineer_features(df)
    assert engineered.loc['AAPL', 'has_momentum_history'] == 0


# ── Edge case / outlier handling tests ───────────────────────────────────────

def make_edge_funds(index: list[str], pe_ratios, debt_to_equities,
                    revenue_growths, dividends, roes) -> pd.DataFrame:
    """
    Helper to build a fundamentals-like DataFrame with full technical columns
    already present so merge_features is not needed for unit tests.
    """
    n = len(index)
    return pd.DataFrame({
        'pe_ratio':       pe_ratios,
        'pb_ratio':       [3.5]    * n,
        'roe':            roes,
        'debt_to_equity': debt_to_equities,
        'revenue_growth': revenue_growths,
        'dividend_yield': dividends,
        'beta':           [1.1]    * n,
        'market_cap':     [2e12]   * n,
        'eps_ttm':        [5.0]    * n,
        'as_of_date':     ['2024-01-01'] * n,
        'sector':         ['Technology'] * n,
        # technical cols added so _engineer_features works standalone
        'momentum_3m':    [0.05]   * n,
        'momentum_6m':    [0.10]   * n,
        'volatility':     [0.20]   * n,
        'rsi':            [55.0]   * n,
    }, index=index)


def test_negative_equity_clipped():
    """Negative D/E should be clipped to -5.0 regardless of ticker."""
    from app.features.fundamentals import _clip_outliers, _engineer_features
    df        = make_edge_funds(['X'], [25.0], [-25.0], [0.14], [0.0], [0.22])
    engineered = _engineer_features(df)
    clipped   = _clip_outliers(engineered)
    assert clipped.loc['X', 'debt_to_equity'] >= -5.0


def test_extreme_pe_clipped():
    """PE ratio above 200 should be clipped to 200 regardless of ticker."""
    from app.features.fundamentals import _clip_outliers, _engineer_features
    df        = make_edge_funds(['X'], [367.0], [0.5], [1.14], [0.0], [0.40])
    engineered = _engineer_features(df)
    clipped   = _clip_outliers(engineered)
    assert clipped.loc['X', 'pe_ratio'] <= 200.0


def test_loss_making_not_profitable():
    """Loss-making ticker (negative PE) should have is_profitable=0."""
    from app.features.fundamentals import _engineer_features
    df        = make_edge_funds(['X'], [-5.0], [0.4], [0.06], [0.0], [-0.5])
    engineered = _engineer_features(df)
    assert engineered.loc['X', 'is_profitable'] == 0


def test_full_pipeline_with_edge_cases():
    """Full pipeline should complete without errors on edge case data."""
    df = make_edge_funds(
        index            = ['A', 'B', 'C'],
        pe_ratios        = [-5.0,  367.0,  25.0],
        debt_to_equities = [-25.0, 0.5,    0.3],
        revenue_growths  = [0.06,  1.14,   0.14],
        dividends        = [0.0,   0.0,    0.025],
        roes             = [-0.5,  0.40,   0.22],
    )
    scaled, scaler, imputer = scale_features(df)
    assert not scaled.isnull().any().any()
    assert scaled.shape[0] == len(df)