import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from app.evaluation.backtester import backtest_optimizer, compute_portfolio_metrics

TICKERS = ['AAPL', 'MSFT', 'JNJ', 'XOM']


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def rich_prices():
    """
    Generate 2 years of fake price data (504 trading days).
    Needs to be long enough for walk-forward with 12m train + 3m test.
    """
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=504, freq='B')
    data  = {}
    for ticker in TICKERS:
        start   = np.random.uniform(100, 400)
        returns = np.random.normal(0.0004, 0.018, 504)
        data[ticker] = start * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_weights():
    return {'AAPL': 0.30, 'MSFT': 0.40, 'JNJ': 0.20, 'XOM': 0.10}


# ── backtest_optimizer tests ───────────────────────────────────────────────────

def test_backtest_returns_summary_keys(rich_prices):
    """backtest_optimizer should return summary and periods keys"""
    result = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    assert 'summary' in result
    assert 'periods' in result
    assert 'tradeoffs' in result


def test_backtest_summary_has_expected_fields(rich_prices):
    """summary block should contain all required metrics"""
    result  = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    summary = result['summary']
    required = [
        'periods_tested',
        'avg_optimized_return',
        'avg_equal_weight_return',
        'avg_outperformance',
        'win_rate_vs_equal',
        'best_period',
        'worst_period',
    ]
    for key in required:
        assert key in summary, f"Missing key: {key}"


def test_backtest_periods_is_list(rich_prices):
    """periods should be a list of dicts"""
    result = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    assert isinstance(result['periods'], list)
    assert len(result['periods']) > 0


def test_backtest_period_has_expected_fields(rich_prices):
    """each period dict should have required fields"""
    result = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    period = result['periods'][0]
    required = [
        'period_start',
        'period_end',
        'optimized_return',
        'equal_weight_return',
        'outperformance',
    ]
    for key in required:
        assert key in period, f"Missing key in period: {key}"


def test_backtest_win_rate_between_0_and_1(rich_prices):
    """win rate should be a valid probability"""
    result   = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    win_rate = result['summary']['win_rate_vs_equal']
    assert 0.0 <= win_rate <= 1.0


def test_backtest_outperformance_matches_returns(rich_prices):
    """outperformance should equal optimized - equal weight return"""
    result = backtest_optimizer(rich_prices, TICKERS, risk='moderate')
    for period in result['periods']:
        expected = round(
            period['optimized_return'] - period['equal_weight_return'], 4
        )
        assert abs(period['outperformance'] - expected) < 1e-6


def test_backtest_handles_invalid_tickers(rich_prices):
    """backtest should skip tickers not in prices, not crash"""
    mixed = ['AAPL', 'MSFT', 'FAKE_TICKER']
    result = backtest_optimizer(rich_prices, mixed, risk='moderate')
    assert 'summary' in result


def test_backtest_insufficient_data():
    """backtest should return error when data is too short"""
    dates  = pd.date_range('2024-01-01', periods=50, freq='B')
    prices = pd.DataFrame(
        {t: np.random.uniform(100, 200, 50) for t in TICKERS},
        index=dates
    )
    result = backtest_optimizer(prices, TICKERS)
    assert 'error' in result


def test_backtest_tradeoffs_block(rich_prices):
    """tradeoffs block should document methodology"""
    result = backtest_optimizer(rich_prices, TICKERS)
    assert result['tradeoffs']['method'] == 'walk_forward'
    assert 'train_months' in result['tradeoffs']
    assert 'note' in result['tradeoffs']


def test_backtest_conservative_vs_aggressive(rich_prices):
    """conservative and aggressive should produce different results"""
    conservative = backtest_optimizer(rich_prices, TICKERS, risk='conservative')
    aggressive   = backtest_optimizer(rich_prices, TICKERS, risk='aggressive')
    assert (
        conservative['summary']['avg_optimized_return'] !=
        aggressive['summary']['avg_optimized_return']
    )


# ── compute_portfolio_metrics tests ───────────────────────────────────────────

def test_metrics_returns_expected_keys(rich_prices, sample_weights):
    """compute_portfolio_metrics should return all required keys"""
    result   = compute_portfolio_metrics(rich_prices, sample_weights)
    required = [
        'realized_annual_return',
        'realized_volatility',
        'realized_sharpe',
        'max_drawdown',
        'total_return',
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_metrics_volatility_positive(rich_prices, sample_weights):
    """volatility should always be positive"""
    result = compute_portfolio_metrics(rich_prices, sample_weights)
    assert result['realized_volatility'] > 0


def test_metrics_max_drawdown_non_positive(rich_prices, sample_weights):
    """max drawdown should be <= 0"""
    result = compute_portfolio_metrics(rich_prices, sample_weights)
    assert result['max_drawdown'] <= 0


def test_metrics_sharpe_is_float(rich_prices, sample_weights):
    """sharpe ratio should be a float"""
    result = compute_portfolio_metrics(rich_prices, sample_weights)
    assert isinstance(result['realized_sharpe'], float)


def test_metrics_skips_unknown_tickers(rich_prices):
    """metrics should handle weights with tickers not in prices"""
    weights = {'AAPL': 0.5, 'FAKE': 0.3, 'MSFT': 0.2}
    result  = compute_portfolio_metrics(rich_prices, weights)
    assert 'realized_annual_return' in result


def test_metrics_equal_weight_benchmark(rich_prices):
    """equal weight portfolio should return sensible metrics"""
    n       = len(TICKERS)
    weights = {t: 1/n for t in TICKERS}
    result  = compute_portfolio_metrics(rich_prices, weights)
    assert result['realized_volatility'] > 0
    assert -1.0 < result['total_return'] < 10.0  # sanity bounds