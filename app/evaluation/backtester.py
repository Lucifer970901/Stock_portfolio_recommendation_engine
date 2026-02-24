
import pandas as pd
import numpy as np
from app.core.logger import get_logger
from app.models.optimizer import optimize_portfolio

log = get_logger(__name__)

def backtest_optimizer(
    prices:        pd.DataFrame,
    tickers:       list[str],
    risk:          str = 'moderate',
    train_months:  int = 12,
    test_months:   int = 3,
) -> dict:
    """
    Walk-forward validation of portfolio optimizer.

    Methodology:
    - Train on `train_months` of data
    - Test on next `test_months` (no lookahead)
    - Roll forward by test_months and repeat
    - Compare optimized weights vs equal weight baseline

    Tradeoff vs simple train/test split:
    Walk-forward is more conservative and realistic but needs
    more data and is slower to compute.
    """
    valid    = [t for t in tickers if t in prices.columns]
    prices   = prices[valid].dropna()
    train_d  = train_months * 21
    test_d   = test_months  * 21
    results  = []

    log.info(f"Starting walk-forward backtest for {valid}, periods={len(prices)}")

    for i in range(train_d, len(prices) - test_d, test_d):
        train = prices.iloc[i - train_d : i]
        test  = prices.iloc[i : i + test_d]

        try:
            opt    = optimize_portfolio(valid, train, risk)
            weights = opt['weights']
        except Exception as e:
            log.warning(f"Optimization failed at step {i}: {e}")
            continue

        # Optimized portfolio return
        opt_return = sum(
            weights.get(t, 0) * (test[t].iloc[-1] / test[t].iloc[0] - 1)
            for t in valid if t in test.columns
        )

        # Equal weight baseline
        eq_return = np.mean([
            test[t].iloc[-1] / test[t].iloc[0] - 1
            for t in valid if t in test.columns
        ])

        results.append({
            'period_start':        prices.index[i].strftime('%Y-%m-%d'),
            'period_end':          prices.index[min(i+test_d, len(prices)-1)].strftime('%Y-%m-%d'),
            'optimized_return':    round(opt_return,  4),
            'equal_weight_return': round(eq_return,   4),
            'outperformance':      round(opt_return - eq_return, 4),
            'predicted_sharpe':    opt['sharpe_ratio'],
            'top_weight':          max(weights, key=weights.get) if weights else None,
        })

    df = pd.DataFrame(results)

    if df.empty:
        return {'error': 'Not enough data for backtesting'}

    return {
        'summary': {
            'periods_tested':          len(df),
            'avg_optimized_return':    round(df['optimized_return'].mean(),    4),
            'avg_equal_weight_return': round(df['equal_weight_return'].mean(), 4),
            'avg_outperformance':      round(df['outperformance'].mean(),      4),
            'win_rate_vs_equal':       round((df['outperformance'] > 0).mean(),4),
            'best_period':             df.loc[df['optimized_return'].idxmax(), 'period_start'],
            'worst_period':            df.loc[df['optimized_return'].idxmin(), 'period_start'],
        },
        'periods': df.to_dict(orient='records'),
        'tradeoffs': {
            'method':      'walk_forward',
            'train_months': train_months,
            'test_months':  test_months,
            'note':        'Walk-forward prevents lookahead bias. Simple hold-out would overestimate performance.'
        }
    }

def compute_portfolio_metrics(
    prices:  pd.DataFrame,
    weights: dict[str, float],
    risk_free_rate: float = 0.05
) -> dict:
    """
    Compute realized portfolio metrics from actual price history.
    Compare predicted vs realized to measure optimizer accuracy.
    """
    valid   = {t: w for t, w in weights.items() if t in prices.columns}
    returns = prices[list(valid.keys())].pct_change().dropna()

    port_returns = sum(w * returns[t] for t, w in valid.items())

    ann_return = port_returns.mean() * 252
    ann_vol    = port_returns.std()  * np.sqrt(252)
    sharpe     = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    max_dd     = (port_returns.cumsum() - port_returns.cumsum().cummax()).min()

    return {
        'realized_annual_return': round(ann_return, 4),
        'realized_volatility':    round(ann_vol,    4),
        'realized_sharpe':        round(sharpe,     4),
        'max_drawdown':           round(max_dd,     4),
        'total_return':           round(port_returns.sum(), 4),
    }