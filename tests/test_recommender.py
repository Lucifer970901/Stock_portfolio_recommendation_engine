"""
Tests for similarity, clustering, and optimizer models.

All tests are self-contained with inline fixtures.
No real API calls are made — all data is synthetic.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.models.similarity import (
    build_similarity_matrix,
    build_similarity_matrices,
    get_similar_stocks,
    get_complementary_stocks,
)
from app.models.clustering import cluster_stocks
from app.models.optimizer import optimize_portfolio

TICKERS = ['AAPL', 'MSFT', 'JNJ', 'XOM', 'JPM']


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_scaled():
    """Scaled feature DataFrame with realistic column names."""
    from app.features.fundamentals import FUNDAMENTAL_COLS, TECHNICAL_COLS, ENGINEERED_COLS
    all_cols = FUNDAMENTAL_COLS + TECHNICAL_COLS + ENGINEERED_COLS
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(len(TICKERS), len(all_cols)),
        index=TICKERS,
        columns=all_cols,
    )


@pytest.fixture
def sample_combined():
    """Raw fundamentals + technical DataFrame with all expected columns."""
    return pd.DataFrame({
        'pe_ratio':        [28.0,  35.0,  21.0,  15.0,  12.0],
        'pb_ratio':        [3.5,   12.0,  3.8,   1.8,   1.5],
        'roe':             [0.25,  0.40,  0.18,  0.12,  0.14],
        'debt_to_equity':  [1.2,   0.5,   0.6,   0.3,   1.8],
        'revenue_growth':  [0.06,  0.15,  0.04,  0.01,  0.09],
        'dividend_yield':  [0.005, 0.0,   0.03,  0.04,  0.02],
        'beta':            [1.2,   0.9,   0.7,   0.8,   1.1],
        'market_cap':      [3e12,  2.8e12, 4e11, 5e11,  4.5e11],
        'eps_ttm':         [8.0,   16.0,  11.0,  7.0,   4.0],
        'volatility':      [0.22,  0.24,  0.19,  0.25,  0.28],
        'momentum_3m':     [0.05,  -0.10, 0.02,  0.08,  -0.03],
        'momentum_6m':     [0.10,  -0.20, 0.04,  0.15,  -0.06],
        'rsi':             [55.0,  45.0,  52.0,  60.0,  48.0],
        'sector':          ['Technology', 'Technology', 'Healthcare',
                            'Energy', 'Financial Services'],
        'cluster_label':   ['Quality Growth', 'Quality Growth', 'Defensive Income',
                            'Energy', 'Financials'],
        'as_of_date':      ['2024-01-01'] * 5,
    }, index=TICKERS)


@pytest.fixture
def sample_prices():
    """504 days of synthetic price data (2 years of trading days)."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=504, freq='B')
    data  = {
        t: np.cumprod(1 + np.random.normal(0.0003, 0.015, 504)) * 100
        for t in TICKERS
    }
    return pd.DataFrame(data, index=dates)


# ── Similarity matrix tests ───────────────────────────────────────────────────

def test_similarity_matrix_shape(sample_scaled):
    """Combined similarity matrix should be square with correct dimensions."""
    matrix = build_similarity_matrix(sample_scaled)
    n = len(sample_scaled)
    assert matrix.shape == (n, n)


def test_similarity_matrix_diagonal(sample_scaled):
    """Diagonal should be 1.0 — each stock is identical to itself."""
    matrix = build_similarity_matrix(sample_scaled)
    for ticker in sample_scaled.index:
        assert abs(matrix.loc[ticker, ticker] - 1.0) < 1e-6


def test_similarity_matrix_symmetric(sample_scaled):
    """Similarity matrix should be symmetric."""
    matrix = build_similarity_matrix(sample_scaled)
    assert (matrix - matrix.T).abs().max().max() < 1e-6


def test_similarity_values_in_valid_range(sample_scaled):
    """All cosine similarity values should be in [-1, 1]."""
    matrix = build_similarity_matrix(sample_scaled)
    assert matrix.values.min() >= -1.01
    assert matrix.values.max() <=  1.01


def test_build_similarity_matrices_returns_all_keys(sample_scaled):
    """build_similarity_matrices should return fundamental, technical, combined."""
    mats = build_similarity_matrices(sample_scaled)
    assert 'fundamental' in mats
    assert 'technical'   in mats
    assert 'combined'    in mats


def test_build_similarity_matrices_combined_is_blend(sample_scaled):
    """Combined matrix should be between fundamental and technical matrices."""
    mats = build_similarity_matrices(sample_scaled)
    if mats['fundamental'] is not None and mats['technical'] is not None:
        # Combined should not be identical to either component
        assert not mats['combined'].equals(mats['fundamental'])
        assert not mats['combined'].equals(mats['technical'])


# ── get_similar_stocks tests ──────────────────────────────────────────────────

def test_get_similar_returns_correct_n(sample_scaled, sample_combined):
    """get_similar_stocks should return exactly top_n results."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert len(result) == 3


def test_get_similar_excludes_self(sample_scaled, sample_combined):
    """get_similar_stocks should not include the query ticker."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert 'AAPL' not in result.index


def test_get_similar_sorted_descending(sample_scaled, sample_combined):
    """Results should be sorted by similarity descending."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=4)
    sims   = result['similarity'].tolist()
    assert sims == sorted(sims, reverse=True)


def test_get_similar_invalid_ticker(sample_scaled, sample_combined):
    """get_similar_stocks should raise ValueError for unknown ticker."""
    matrix = build_similarity_matrix(sample_scaled)
    with pytest.raises(ValueError):
        get_similar_stocks('FAKE', matrix, sample_combined)


def test_get_similar_same_cluster_filter(sample_scaled, sample_combined):
    """same_cluster=True should restrict results to same cluster_label."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks(
        'AAPL', matrix, sample_combined, top_n=4, same_cluster=True
    )
    for ticker in result.index:
        assert sample_combined.loc[ticker, 'cluster_label'] == 'Quality Growth'


def test_get_similar_has_similarity_column(sample_scaled, sample_combined):
    """Results should include a similarity column."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert 'similarity' in result.columns


# ── get_complementary_stocks tests ───────────────────────────────────────────

def test_get_complementary_returns_correct_n(sample_scaled, sample_combined):
    """get_complementary_stocks should return top_n results."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_complementary_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert len(result) == 3


def test_get_complementary_excludes_self(sample_scaled, sample_combined):
    """get_complementary_stocks should not include the query ticker."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_complementary_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert 'AAPL' not in result.index


def test_get_complementary_sorted_ascending(sample_scaled, sample_combined):
    """Complementary results should be sorted by similarity ascending."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_complementary_stocks('AAPL', matrix, sample_combined, top_n=3)
    sims   = result['similarity'].tolist()
    assert sims == sorted(sims)


def test_get_complementary_excludes_same_cluster(sample_scaled, sample_combined):
    """exclude_same_cluster=True should skip tickers in same cluster."""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_complementary_stocks(
        'AAPL', matrix, sample_combined, top_n=4,
        exclude_same_cluster=True,
    )
    for ticker in result.index:
        assert sample_combined.loc[ticker, 'cluster_label'] != 'Quality Growth'


def test_get_complementary_invalid_ticker(sample_scaled, sample_combined):
    """get_complementary_stocks should raise ValueError for unknown ticker."""
    matrix = build_similarity_matrix(sample_scaled)
    with pytest.raises(ValueError):
        get_complementary_stocks('FAKE', matrix, sample_combined)


# ── cluster_stocks tests ──────────────────────────────────────────────────────

def test_cluster_adds_columns(sample_scaled, sample_combined):
    """cluster_stocks should add cluster and cluster_label columns."""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    assert 'cluster'       in result.columns
    assert 'cluster_label' in result.columns


def test_cluster_correct_n_unique(sample_scaled, sample_combined):
    """Number of unique KMeans clusters should equal n_clusters."""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    assert result['cluster'].nunique() == 2


def test_cluster_preserves_all_rows(sample_scaled, sample_combined):
    """Clustering should not drop any tickers."""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    assert len(result) == len(sample_combined)


def test_cluster_label_distressed(sample_scaled, sample_combined):
    """Tickers with negative EPS should be labelled Distressed."""
    combined = sample_combined.copy()
    combined.loc['AAPL', 'eps_ttm'] = -1.0
    result = cluster_stocks(sample_scaled, combined, n_clusters=2)
    assert result.loc['AAPL', 'cluster_label'] == 'Distressed'


def test_cluster_label_negative_equity(sample_scaled, sample_combined):
    """Tickers with D/E < -3 should be labelled Negative Equity."""
    combined = sample_combined.copy()
    combined.loc['XOM', 'debt_to_equity'] = -10.0
    result = cluster_stocks(sample_scaled, combined, n_clusters=2)
    assert result.loc['XOM', 'cluster_label'] == 'Negative Equity'


def test_cluster_label_hypergrowth(sample_scaled, sample_combined):
    """Tickers with revenue_growth > 0.5 should be labelled Hypergrowth."""
    combined = sample_combined.copy()
    combined.loc['MSFT', 'revenue_growth'] = 1.14
    result = cluster_stocks(sample_scaled, combined, n_clusters=2)
    assert result.loc['MSFT', 'cluster_label'] == 'Hypergrowth'


def test_cluster_label_sector_fallback(sample_scaled, sample_combined):
    """Normal tickers should fall back to sector-based label."""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    # JPM is Financial Services with normal fundamentals
    assert result.loc['JPM', 'cluster_label'] == 'Financials'


# ── optimize_portfolio tests ──────────────────────────────────────────────────

def test_optimize_returns_all_keys(sample_prices):
    """optimize_portfolio should return all expected output keys."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate')
    for key in [
        'weights', 'allocation', 'leftover_cash',
        'expected_return', 'volatility', 'sharpe_ratio',
        'capital', 'risk_profile', 'n_positions',
    ]:
        assert key in result


def test_optimize_weights_sum_to_one(sample_prices):
    """Portfolio weights should sum to approximately 1.0."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate')
    total  = sum(result['weights'].values())
    assert abs(total - 1.0) < 0.01


def test_optimize_conservative_lower_vol(sample_prices):
    """Conservative should have lower volatility than aggressive."""
    conservative = optimize_portfolio(TICKERS, sample_prices, risk='conservative')
    aggressive   = optimize_portfolio(TICKERS, sample_prices, risk='aggressive')
    assert conservative['volatility'] <= aggressive['volatility']


def test_optimize_skips_invalid_tickers(sample_prices):
    """optimize_portfolio should skip tickers not in prices DataFrame."""
    result = optimize_portfolio(
        ['AAPL', 'MSFT', 'JNJ', 'FAKE_TICKER'], sample_prices, risk='moderate'
    )
    assert 'FAKE_TICKER' not in result['weights']


def test_optimize_allocation_matches_weights(sample_prices):
    """Allocation dict should contain same tickers as weights."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate')
    assert set(result['allocation'].keys()) == set(result['weights'].keys())


def test_optimize_allocation_has_correct_keys(sample_prices):
    """Each allocation entry should have shares, price, allocation, weight."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate')
    for ticker, details in result['allocation'].items():
        for key in ['shares', 'price', 'allocation', 'weight']:
            assert key in details, f"Missing '{key}' in allocation for {ticker}"


def test_optimize_capital_param(sample_prices):
    """Capital parameter should be reflected in result."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate', capital=50000)
    assert result['capital'] == 50000


def test_optimize_risk_profile_in_result(sample_prices):
    """Result should include risk profile name."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='conservative')
    assert result['risk_profile'] == 'conservative'


def test_optimize_n_positions_correct(sample_prices):
    """n_positions should match number of active weights."""
    result = optimize_portfolio(TICKERS, sample_prices, risk='moderate')
    assert result['n_positions'] == len(result['weights'])


def test_optimize_too_few_tickers_raises(sample_prices):
    """optimize_portfolio should raise ValueError with fewer than 2 valid tickers."""
    with pytest.raises(ValueError):
        optimize_portfolio(['AAPL'], sample_prices, risk='moderate')


def test_optimize_all_risk_profiles(sample_prices):
    """All three risk profiles should complete without error."""
    for risk in ['conservative', 'moderate', 'aggressive']:
        result = optimize_portfolio(TICKERS, sample_prices, risk=risk)
        assert result['expected_return'] is not None
        assert result['sharpe_ratio'] is not None