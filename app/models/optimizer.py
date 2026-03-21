"""
Portfolio Optimizer
-------------------
Optimizes portfolio weights using Modern Portfolio Theory via PyPortfolioOpt.

Key design decisions:
  - CAPM expected returns   : more stable than mean historical, anchored to market
  - Ledoit-Wolf shrinkage   : well-conditioned covariance matrix for 50 tickers
  - Three risk profiles     : conservative (min vol), moderate (max sharpe),
                              aggressive (max quadratic utility)
  - Capital allocation      : converts weights to dollar amounts
  - Two-stage fallback      : relaxes bounds then strategy if optimization fails
"""

import pandas as pd
import numpy as np
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    DiscreteAllocation,
)
from pypfopt.exceptions import OptimizationError
from app.core.logger import get_logger
from app.core.config import settings

log = get_logger(__name__)

# Weight bounds per risk profile (min, max allocation per ticker)
WEIGHT_BOUNDS = {
    'conservative': (0.0, 0.10),
    'moderate':     (0.0, 0.20),
    'aggressive':   (0.0, 0.35),
}

# Minimum number of positions per risk profile
MIN_POSITIONS = {
    'conservative': 10,
    'moderate':     6,
    'aggressive':   5,
}

# CAPM market premium assumption (annualised)
MARKET_RISK_PREMIUM = 0.0523   # historical US equity premium ~5.23%
RISK_FREE_RATE      = 0.0525   # approximate current 10Y Treasury yield
RISK_FREE_RATE      = 0.0525   # approximate current 10Y Treasury yield


def _expected_returns_capm(
    prices: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    market_risk_premium: float = MARKET_RISK_PREMIUM,
) -> pd.Series:
    """
    Compute blended expected returns: 50% CAPM + 50% exponentially weighted.

    Pure CAPM with high risk-free rate (5.25%) makes all stocks look similar
    (E(R) = Rf + beta * 5.23%), preventing the optimizer from concentrating.
    Blending with EW historical returns adds cross-sectional differentiation
    while keeping CAPM as a stabilising anchor.

    Args:
        prices:              closing price DataFrame
        risk_free_rate:      annual risk-free rate
        market_risk_premium: expected excess return of market over risk-free

    Returns:
        pd.Series of annualised blended expected returns per ticker
    """
    capm_mu = None
    ew_mu   = None

    # CAPM component
    try:
        capm_mu = expected_returns.capm_return(
            prices,
            risk_free_rate=risk_free_rate,
            frequency=252,
        )
        log.info("CAPM returns computed")
    except Exception as e:
        log.warning(f"CAPM failed ({e})")

    # Exponentially weighted historical returns (recent data weighted more)
    try:
        ew_mu = expected_returns.ema_historical_return(
            prices,
            frequency=252,
            span=252,   # ~1 year half-life
        )
        log.info("EW historical returns computed")
    except Exception as e:
        log.warning(f"EW historical returns failed ({e})")

    # Blend: 50% CAPM + 50% EW, fall back to whichever is available
    if capm_mu is not None and ew_mu is not None:
        mu = 0.5 * capm_mu + 0.5 * ew_mu
        log.info("Using blended CAPM + EW expected returns (50/50)")
    elif capm_mu is not None:
        mu = capm_mu
        log.info("Using CAPM expected returns only")
    elif ew_mu is not None:
        mu = ew_mu
        log.info("Using EW historical returns only")
    else:
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        log.info("Using mean historical returns (fallback)")

    return mu


def _covariance(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ledoit-Wolf shrinkage covariance matrix.

    With 50 tickers, sample covariance is noisy (needs 500+ obs to be
    well-conditioned). Ledoit-Wolf shrinkage regularises it by pulling
    extreme off-diagonal entries toward zero.

    Args:
        prices: closing price DataFrame

    Returns:
        Shrinkage covariance matrix as DataFrame
    """
    try:
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        log.info("Using Ledoit-Wolf shrinkage covariance")
        return cov
    except Exception as e:
        log.warning(f"Ledoit-Wolf failed ({e}), falling back to sample covariance")
        return risk_models.sample_cov(prices, frequency=252)


def _run_optimization(
    ef: EfficientFrontier,
    risk: str,
) -> EfficientFrontier:
    """
    Run the appropriate optimization strategy for the risk profile.
    Raises OptimizationError if the strategy fails.
    """
    if risk == 'conservative':
        ef.min_volatility()
    elif risk == 'aggressive':
        ef.max_quadratic_utility(risk_aversion=2)
    else:
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    return ef


def _allocate_capital(
    weights: dict[str, float],
    latest_prices: pd.Series,
    capital: float,
) -> dict[str, dict]:
    """
    Convert portfolio weights to discrete share counts and dollar amounts.

    Uses PyPortfolioOpt DiscreteAllocation which solves a greedy
    rounding problem to allocate whole shares optimally.

    Args:
        weights:       cleaned portfolio weights dict
        latest_prices: most recent closing prices per ticker
        capital:       total capital in dollars

    Returns:
        dict mapping ticker -> {shares, price, allocation, weight}
    """
    try:
        da = DiscreteAllocation(
            weights,
            latest_prices[list(weights.keys())],
            total_portfolio_value=capital,
        )
        allocation, leftover = da.greedy_portfolio()
        log.info(f"Capital allocated — invested: ${capital - leftover:,.0f}, "
                 f"leftover: ${leftover:,.0f}")

        result = {}
        for ticker, shares in allocation.items():
            price      = float(latest_prices[ticker])
            dollar_amt = round(shares * price, 2)
            result[ticker] = {
                'shares':     shares,
                'price':      round(price, 2),
                'allocation': dollar_amt,
                'weight':     round(weights.get(ticker, 0), 4),
            }
        return result, round(leftover, 2)

    except Exception as e:
        log.warning(f"Discrete allocation failed ({e}), using proportional allocation")
        result = {}
        for ticker, weight in weights.items():
            price      = float(latest_prices.get(ticker, 0))
            dollar_amt = round(weight * capital, 2)
            shares     = int(dollar_amt / price) if price > 0 else 0
            result[ticker] = {
                'shares':     shares,
                'price':      round(price, 2),
                'allocation': dollar_amt,
                'weight':     round(weight, 4),
            }
        return result, 0.0


def optimize_portfolio(
    tickers: list[str],
    prices: pd.DataFrame,
    risk: str = 'moderate',
    capital: float | None = None,
) -> dict:
    """
    Optimize portfolio weights and allocate capital.

    Args:
        tickers: list of candidate ticker symbols
        prices:  historical closing price DataFrame (from fetch_prices)
        risk:    risk profile — 'conservative', 'moderate', or 'aggressive'
        capital: total capital in dollars (defaults to settings.default_capital)

    Returns:
        dict with keys:
          weights          : {ticker: weight} for positions > 1%
          allocation       : {ticker: {shares, price, allocation, weight}}
          leftover_cash    : uninvested cash after discrete allocation
          expected_return  : annualised expected portfolio return
          volatility       : annualised portfolio volatility
          sharpe_ratio     : Sharpe ratio at risk_free_rate
          capital          : total capital used
          risk_profile     : risk profile name
          n_positions      : number of non-zero positions
    """
    capital = capital or settings.default_capital
    risk    = risk if risk in WEIGHT_BOUNDS else 'moderate'

    # Filter to valid tickers with sufficient price history
    valid = [t for t in tickers if t in prices.columns]
    p     = prices[valid].dropna()

    if len(p.columns) < 2:
        raise ValueError(f"Need at least 2 valid tickers, got {len(p.columns)}")
    if len(p) < 60:
        raise ValueError(f"Need at least 60 days of price history, got {len(p)}")

    log.info(
        f"Optimizing portfolio — tickers: {len(valid)}, "
        f"days: {len(p)}, risk: {risk}, capital: ${capital:,.0f}"
    )

    mu  = _expected_returns_capm(p)
    cov = _covariance(p)

    bounds = WEIGHT_BOUNDS[risk]

    # Stage 1: optimize with requested bounds and strategy
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        ef = _run_optimization(ef, risk)

    except OptimizationError:
        log.warning(f"{risk} optimization failed with bounds {bounds}, "
                    f"trying min_volatility")
        # Stage 2: fallback to min_volatility with same bounds
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
            ef.min_volatility()

        except OptimizationError:
            log.warning("min_volatility failed, relaxing weight bounds to (0, 1)")
            # Stage 3: relax bounds entirely
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef.min_volatility()

    weights          = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(
        risk_free_rate=RISK_FREE_RATE, verbose=False
    )

    # Filter to meaningful positions
    active_weights = {k: v for k, v in weights.items() if v > 0.01}

    # Warn if too few positions for the risk profile
    min_pos = MIN_POSITIONS.get(risk, 5)
    if len(active_weights) < min_pos:
        log.warning(
            f"{risk} portfolio has only {len(active_weights)} positions "
            f"(min recommended: {min_pos}) — consider adding more tickers"
        )

    # Capital allocation
    latest_prices  = p.iloc[-1]
    allocation, leftover = _allocate_capital(active_weights, latest_prices, capital)

    log.info(
        f"Portfolio optimised — return: {ret:.2%}, vol: {vol:.2%}, "
        f"sharpe: {sharpe:.2f}, positions: {len(active_weights)}"
    )

    return {
        'weights':          {k: round(v, 4) for k, v in active_weights.items()},
        'allocation':       allocation,
        'leftover_cash':    leftover,
        'expected_return':  round(ret, 4),
        'volatility':       round(vol, 4),
        'sharpe_ratio':     round(sharpe, 4),
        'capital':          capital,
        'risk_profile':     risk,
        'n_positions':      len(active_weights),
    }