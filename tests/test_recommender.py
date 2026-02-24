import pytest
import pandas as pd
import numpy as np
from app.models.similarity import build_similarity_matrix, get_similar_stocks
from app.models.clustering import cluster_stocks, CLUSTER_LABELS
from app.models.optimizer import optimize_portfolio

# ── Similarity tests ───────────────────────────────────────────────────────────

def test_similarity_matrix_shape(sample_scaled):
    """Similarity matrix should be square"""
    matrix = build_similarity_matrix(sample_scaled)
    n = len(sample_scaled)
    assert matrix.shape == (n, n)

def test_similarity_matrix_diagonal(sample_scaled):
    """Diagonal should be 1.0 (stock is identical to itself)"""
    matrix = build_similarity_matrix(sample_scaled)
    for ticker in sample_scaled.index:
        assert abs(matrix.loc[ticker, ticker] - 1.0) < 1e-6

def test_similarity_matrix_symmetric(sample_scaled):
    """Similarity matrix should be symmetric"""
    matrix = build_similarity_matrix(sample_scaled)
    assert (matrix - matrix.T).abs().max().max() < 1e-6

def test_get_similar_stocks_returns_correct_n(sample_scaled, sample_combined):
    """get_similar_stocks should return exactly top_n results"""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert len(result) == 3

def test_get_similar_stocks_excludes_self(sample_scaled, sample_combined):
    """get_similar_stocks should not include the query ticker"""
    matrix = build_similarity_matrix(sample_scaled)
    result = get_similar_stocks('AAPL', matrix, sample_combined, top_n=3)
    assert 'AAPL' not in result.index

def test_get_similar_stocks_invalid_ticker(sample_scaled, sample_combined):
    """get_similar_stocks should raise ValueError for unknown ticker"""
    matrix = build_similarity_matrix(sample_scaled)
    with pytest.raises(ValueError):
        get_similar_stocks('FAKE', matrix, sample_combined)

# ── Clustering tests ───────────────────────────────────────────────────────────

def test_cluster_stocks_adds_columns(sample_scaled, sample_combined):
    """cluster_stocks should add cluster and cluster_label columns"""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    assert 'cluster'       in result.columns
    assert 'cluster_label' in result.columns

def test_cluster_stocks_correct_count(sample_scaled, sample_combined):
    """Number of unique clusters should equal n_clusters"""
    n = 2
    result   = cluster_stocks(sample_scaled, sample_combined, n_clusters=n)
    n_unique = result['cluster'].nunique()
    assert n_unique == n

def test_cluster_stocks_preserves_rows(sample_scaled, sample_combined):
    """Clustering should not drop any rows"""
    result = cluster_stocks(sample_scaled, sample_combined, n_clusters=2)
    assert len(result) == len(sample_combined)

# ── Optimizer tests ────────────────────────────────────────────────────────────

def test_optimize_returns_expected_keys(sample_prices):
    """optimize_portfolio should return all expected keys"""
    result = optimize_portfolio(
        ['AAPL', 'MSFT', 'JNJ', 'XOM'], sample_prices, risk='moderate'
    )
    assert 'weights'         in result
    assert 'expected_return' in result
    assert 'volatility'      in result
    assert 'sharpe_ratio'    in result

def test_optimize_weights_sum_to_one(sample_prices):
    """Portfolio weights should sum to approximately 1.0"""
    result  = optimize_portfolio(
        ['AAPL', 'MSFT', 'JNJ', 'XOM'], sample_prices, risk='moderate'
    )
    total = sum(result['weights'].values())
    assert abs(total - 1.0) < 0.01

def test_optimize_conservative_lower_volatility(sample_prices):
    """Conservative portfolio should have lower volatility than aggressive"""
    conservative = optimize_portfolio(
        ['AAPL', 'MSFT', 'JNJ', 'XOM'], sample_prices, risk='conservative'
    )
    aggressive = optimize_portfolio(
        ['AAPL', 'MSFT', 'JNJ', 'XOM'], sample_prices, risk='aggressive'
    )
    assert conservative['volatility'] <= aggressive['volatility']

def test_optimize_handles_invalid_tickers(sample_prices):
    """optimize_portfolio should skip tickers not in prices"""
    result = optimize_portfolio(
        ['AAPL', 'FAKE_TICKER'], sample_prices, risk='moderate'
    )
    assert 'FAKE_TICKER' not in result['weights']