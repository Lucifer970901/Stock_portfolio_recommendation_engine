import pandas as pd
import numpy as np
from app.core.logger import get_logger
from app.models.optimizer import optimize_portfolio

log = get_logger(__name__)


def backtest_optimizer(
    prices:          pd.DataFrame,
    tickers:         list[str],
    risk:            str       = 'moderate',
    train_months:    int       = 9,
    test_months:     int       = 3,
    exclude_tickers: list[str] = None,
) -> dict:
    """
    Walk-forward validation of portfolio optimizer.

    Methodology:
    - Train on `train_months` of data
    - Test on next `test_months` (no lookahead)
    - Roll forward by test_months and repeat
    - Compare optimized weights vs equal weight baseline

    Args:
        prices:          historical price DataFrame
        tickers:         candidate ticker symbols
        risk:            risk profile (conservative / moderate / aggressive)
        train_months:    months of training data per window (default 9)
        test_months:     months of test data per window (default 3)
        exclude_tickers: tickers to exclude from optimization
                         (e.g. loss-making / negative equity stocks)
                         Should match recommender.investable_tickers exclusions.

    Returns:
        dict with summary, periods, and tradeoffs
    """
    exclude = set(exclude_tickers or [])

    # Filter to valid + investable tickers
    valid = [
        t for t in tickers
        if t in prices.columns and t not in exclude
    ]

    if exclude:
        excluded_found = [t for t in tickers if t in exclude]
        if excluded_found:
            log.info(f"Excluded from backtest: {excluded_found}")

    prices  = prices[valid].dropna()
    train_d = train_months * 21
    test_d  = test_months  * 21
    results = []

    log.info(
        f"Walk-forward backtest — tickers: {len(valid)}, "
        f"periods: {len(prices)}, risk: {risk}, "
        f"train: {train_months}m, test: {test_months}m"
    )

    if len(prices) < train_d + test_d:
        return {'error': f'Not enough data — need {train_d + test_d} days, got {len(prices)}'}

    for i in range(train_d, len(prices) - test_d, test_d):
        train = prices.iloc[i - train_d : i]
        test  = prices.iloc[i : i + test_d]

        try:
            opt     = optimize_portfolio(valid, train, risk)
            weights = opt['weights']
        except Exception as e:
            log.warning(f"Optimization failed at step {i}: {e}")
            continue

        # Optimized portfolio daily returns in test period
        opt_daily = sum(
            weights.get(t, 0) * test[t].pct_change().dropna()
            for t in valid if t in test.columns
        )

        # Equal weight daily returns
        eq_daily = test[valid].pct_change().dropna().mean(axis=1)

        # Period returns
        opt_return = float((1 + opt_daily).prod() - 1)
        eq_return  = float((1 + eq_daily).prod()  - 1)

        # Per-period max drawdown (optimized portfolio)
        cumulative = (1 + opt_daily).cumprod()
        drawdown   = (cumulative - cumulative.cummax()) / cumulative.cummax()
        period_mdd = float(drawdown.min())

        # Annualized Sharpe for test period
        ann_factor    = np.sqrt(252 / len(opt_daily)) if len(opt_daily) > 1 else 1
        period_sharpe = (
            float(opt_daily.mean() * 252 / (opt_daily.std() * np.sqrt(252)))
            if opt_daily.std() > 0 else 0.0
        )

        results.append({
            'period_start':        prices.index[i].strftime('%Y-%m-%d'),
            'period_end':          prices.index[min(i + test_d, len(prices) - 1)].strftime('%Y-%m-%d'),
            'optimized_return':    round(opt_return,     4),
            'equal_weight_return': round(eq_return,      4),
            'outperformance':      round(opt_return - eq_return, 4),
            'period_max_drawdown': round(period_mdd,     4),
            'period_sharpe':       round(period_sharpe,  4),
            'predicted_sharpe':    opt['sharpe_ratio'],
            'n_positions':         opt['n_positions'],
            'top_weight':          max(weights, key=weights.get) if weights else None,
        })

    df = pd.DataFrame(results)

    if df.empty:
        return {'error': 'Not enough data for backtesting'}

    # Overall realized metrics across all test periods
    avg_opt = df['optimized_return'].mean()
    avg_eq  = df['equal_weight_return'].mean()
    win_rate = (df['outperformance'] > 0).mean()

    return {
        'summary': {
            'periods_tested':          len(df),
            'train_months':            train_months,
            'test_months':             test_months,
            'tickers_used':            len(valid),
            'excluded_tickers':        list(exclude & set(tickers)),
            'avg_optimized_return':    round(avg_opt,  4),
            'avg_equal_weight_return': round(avg_eq,   4),
            'avg_outperformance':      round(avg_opt - avg_eq, 4),
            'win_rate_vs_equal':       round(win_rate, 4),
            'avg_max_drawdown':        round(df['period_max_drawdown'].mean(), 4),
            'worst_drawdown':          round(df['period_max_drawdown'].min(),  4),
            'avg_period_sharpe':       round(df['period_sharpe'].mean(),       4),
            'best_period':             df.loc[df['optimized_return'].idxmax(), 'period_start'],
            'worst_period':            df.loc[df['optimized_return'].idxmin(), 'period_start'],
        },
        'periods': df.to_dict(orient='records'),
        'tradeoffs': {
            'method':       'walk_forward',
            'train_months': train_months,
            'test_months':  test_months,
            'note': (
                'Walk-forward prevents lookahead bias. '
                'Simple hold-out would overestimate performance. '
                f'With {len(prices)} trading days, {len(df)} non-overlapping '
                f'{test_months}-month test windows were evaluated.'
            ),
        },
    }


def compute_portfolio_metrics(
    prices:         pd.DataFrame,
    weights:        dict[str, float],
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute realized portfolio metrics from actual price history.
    Compare predicted vs realized to measure optimizer accuracy.

    Args:
        prices:         historical price DataFrame
        weights:        portfolio weights dict from optimize_portfolio()
        risk_free_rate: annual risk-free rate for Sharpe calculation

    Returns:
        dict with realized return, volatility, Sharpe, drawdown, total return
    """
    valid        = {t: w for t, w in weights.items() if t in prices.columns}
    returns      = prices[list(valid.keys())].pct_change().dropna()
    port_returns = sum(w * returns[t] for t, w in valid.items())

    ann_return = port_returns.mean() * 252
    ann_vol    = port_returns.std()  * np.sqrt(252)
    sharpe     = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown on cumulative returns
    cumulative = (1 + port_returns).cumprod()
    drawdown   = (cumulative - cumulative.cummax()) / cumulative.cummax()
    max_dd     = float(drawdown.min())

    # Calmar ratio: annualized return / abs(max drawdown)
    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0.0

    return {
        'realized_annual_return': round(ann_return,          4),
        'realized_volatility':    round(ann_vol,             4),
        'realized_sharpe':        round(sharpe,              4),
        'max_drawdown':           round(max_dd,              4),
        'calmar_ratio':           round(calmar,              4),
        'total_return':           round(port_returns.sum(),  4),
    }