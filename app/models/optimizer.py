import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
from app.core.logger import get_logger

log = get_logger(__name__)

WEIGHT_BOUNDS = {
    'conservative': (0, 0.10),
    'moderate':     (0, 0.20),
    'aggressive':   (0, 0.35),
}

def optimize_portfolio(tickers: list[str],
                        prices: pd.DataFrame,
                        risk: str = 'moderate') -> dict:
    valid = [t for t in tickers if t in prices.columns]
    p     = prices[valid].dropna()

    mu  = expected_returns.mean_historical_return(p)
    cov = risk_models.sample_cov(p)

    def try_optimize(weight_bounds):
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
        try:
            if risk == 'conservative':
                ef.min_volatility()
            elif risk == 'aggressive':
                ef.max_quadratic_utility()
            else:
                ef.max_sharpe()
            return ef
        except OptimizationError:
            # Fallback 1: try min volatility
            log.warning("max_sharpe failed, falling back to min_volatility")
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
            ef.min_volatility()
            return ef

    # Try with requested bounds first, then relax if needed
    try:
        ef = try_optimize(WEIGHT_BOUNDS.get(risk, (0, 0.20)))
    except OptimizationError:
        # Fallback 2: relax weight bounds entirely
        log.warning("Optimization failed with tight bounds, relaxing to (0, 1)")
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
        ef.min_volatility()

    weights          = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.05, verbose=False)

    log.info(f"Portfolio optimized — return: {ret:.2%}, sharpe: {sharpe:.2f}")

    return {
        'weights':         {k: round(v, 4) for k, v in weights.items() if v > 0.01},
        'expected_return': round(ret, 4),
        'volatility':      round(vol, 4),
        'sharpe_ratio':    round(sharpe, 4),
    }