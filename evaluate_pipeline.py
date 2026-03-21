"""
Full Pipeline Evaluation
------------------------
End-to-end evaluation of the stock recommender pipeline:
  1. Data pipeline health check
  2. Feature pipeline validation
  3. Clustering quality
  4. Similarity sanity check
  5. Walk-forward backtest (all 3 risk profiles)
  6. Realized vs predicted metrics
  7. Summary scorecard

Run with:
    uv run python evaluate_pipeline.py
"""

import sys
import io
# Force UTF-8 output on Windows to avoid charmap encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.data.fetcher import fetch_prices, fetch_fundamentals
from app.features.technical import compute_technical_features
from app.features.fundamentals import merge_features, scale_features
from app.models.clustering import cluster_stocks, get_cluster_stats
from app.models.similarity import build_similarity_matrices, get_similar_stocks
from app.models.optimizer import optimize_portfolio
from app.evaluation.backtester import backtest_optimizer, compute_portfolio_metrics
from app.services.recommender import recommender

log = get_logger(__name__)

SEP  = "=" * 65
SEP2 = "-" * 65


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str):   print(f"  [OK]   {msg}")
def warn(msg: str): print(f"  [WARN] {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")
def info(msg: str): print(f"         {msg}")


# ── 1. Build full pipeline ────────────────────────────────────────────────────

section("1 / DATA PIPELINE")
t0 = time.time()

prices = fetch_prices(settings.tickers, period='5y')
ok(f"Prices fetched — {len(prices.columns)} tickers, {len(prices)} trading days")

funds = fetch_fundamentals(settings.tickers)
ok(f"Fundamentals fetched — {len(funds)} tickers, {len(funds.columns)} fields")

missing_prices = [t for t in settings.tickers if t not in prices.columns]
if missing_prices:
    warn(f"Missing price data: {missing_prices}")
else:
    ok("All tickers have price data")

nan_pct = funds.isna().mean().mean() * 100
if nan_pct > 20:
    warn(f"High NaN rate in fundamentals: {nan_pct:.1f}%")
else:
    ok(f"Fundamentals NaN rate: {nan_pct:.1f}%")

info(f"Data fetch time: {time.time()-t0:.1f}s")


# ── 2. Feature pipeline ───────────────────────────────────────────────────────

section("2 / FEATURE PIPELINE")

tech   = compute_technical_features(prices)
ok(f"Technical features — {len(tech)} tickers, {len(tech.columns)} features")

merged = merge_features(funds, tech)
ok(f"Merged shape: {merged.shape}")

if len(merged) < len(funds) * 0.8:
    warn(f"Inner join dropped {len(funds)-len(merged)} tickers — check price history")
else:
    ok(f"Inner join retained {len(merged)}/{len(funds)} tickers")

scaled, scaler, imputer = scale_features(merged)
ok(f"Scaled shape: {scaled.shape}")

nan_after_scale = scaled.isna().sum().sum()
if nan_after_scale > 0:
    fail(f"NaNs after scaling: {nan_after_scale}")
else:
    ok("No NaNs after scaling")

means = scaled.mean().abs()
if means.max() > 0.5:
    warn(f"Scaling may be off — max abs mean: {means.max():.3f} ({means.idxmax()})")
else:
    ok(f"Feature means near zero (max abs: {means.max():.3f})")


# ── 3. Clustering ─────────────────────────────────────────────────────────────

section("3 / CLUSTERING")

clustered = cluster_stocks(scaled, merged)

label_counts = clustered['cluster_label'].value_counts()
info("Cluster label distribution:")
for label, count in label_counts.items():
    bar = '#' * count
    print(f"      {label:22s} {count:3d}  {bar}")

singleton_clusters = label_counts[label_counts == 1]
if len(singleton_clusters) > 2:
    warn(f"{len(singleton_clusters)} singleton clusters — consider reducing n_clusters")
else:
    ok(f"Cluster sizes look reasonable ({len(singleton_clusters)} singletons)")

dominant = label_counts.iloc[0]
if dominant > len(clustered) * 0.5:
    warn(f"Largest cluster '{label_counts.index[0]}' has {dominant} tickers ({dominant/len(clustered):.0%}) — very broad")
else:
    ok(f"Largest cluster has {dominant} tickers ({dominant/len(clustered):.0%})")

# Show cluster stats
print(f"\n  Cluster fundamentals summary:")
stats = get_cluster_stats(clustered)
print(stats[['pe_ratio', 'revenue_growth', 'beta', 'roe']].to_string(
    float_format=lambda x: f'{x:.3f}'
))


# ── 4. Investable universe ────────────────────────────────────────────────────

section("4 / INVESTABLE UNIVERSE")

recommender.build()
investable = recommender.investable_tickers
excluded   = sorted(set(settings.tickers) - set(investable))

ok(f"Investable tickers: {len(investable)} / {len(settings.tickers)}")
if excluded:
    warn(f"Excluded from optimization: {excluded}")
    for t in excluded:
        row = merged.loc[t] if t in merged.index else None
        if row is not None:
            eps = row.get('eps_ttm', float('nan'))
            de  = row.get('debt_to_equity', float('nan'))
            info(f"  {t}: eps_ttm={eps:.2f}, d/e={de:.2f}, label={clustered.loc[t,'cluster_label'] if t in clustered.index else 'N/A'}")


# ── 5. Similarity sanity check ────────────────────────────────────────────────

section("5 / SIMILARITY SANITY CHECK")

matrices = build_similarity_matrices(scaled)
sim_df   = matrices['combined']

test_pairs = [
    ('AAPL',  'MSFT',  'should be similar — both Quality Growth tech'),
    ('JPM',   'BAC',   'should be similar — both large-cap financials'),
    ('NVDA',  'KO',    'should be dissimilar — hypergrowth vs defensive'),
    ('XOM',   'CVX',   'should be similar — both energy'),
]

for t1, t2, note in test_pairs:
    if t1 not in sim_df.index or t2 not in sim_df.index:
        warn(f"{t1}/{t2} not in universe — skipping")
        continue
    score = sim_df.loc[t1, t2]
    status = ok if (
        ('similar' in note and score > 0.3) or
        ('dissimilar' in note and score < 0.3)
    ) else warn
    status(f"{t1} <-> {t2}: {score:.3f}  ({note})")

# Top 3 similar to AAPL
aapl_similar = get_similar_stocks('AAPL', sim_df, clustered, top_n=3)
info(f"\n  Top 3 similar to AAPL:")
for ticker, row in aapl_similar.iterrows():
    info(f"  {ticker:6s} similarity={row['similarity']:.3f}  label={row.get('cluster_label','?')}")


# ── 6. Walk-forward backtest ──────────────────────────────────────────────────

section("6 / WALK-FORWARD BACKTEST")
info(f"Train: 9 months  |  Test: 3 months  |  Excluded: {excluded}")
print()

backtest_results = {}

for risk in ['conservative', 'moderate', 'aggressive']:
    print(f"  {SEP2}")
    print(f"  {risk.upper()}")
    print(f"  {SEP2}")

    result = backtest_optimizer(
        prices,
        settings.tickers,
        risk=risk,
        train_months=9,
        test_months=3,
        exclude_tickers=excluded,
    )

    if 'error' in result:
        fail(result['error'])
        continue

    backtest_results[risk] = result
    s = result['summary']

    print(f"  Periods tested   : {s['periods_tested']}")
    print(f"  Tickers used     : {s['tickers_used']}")
    print(f"  Avg opt return   : {s['avg_optimized_return']*100:+.2f}%")
    print(f"  Avg eq return    : {s['avg_equal_weight_return']*100:+.2f}%")
    print(f"  Outperformance   : {s['avg_outperformance']*100:+.2f}%")
    print(f"  Win rate         : {s['win_rate_vs_equal']*100:.0f}%")
    print(f"  Avg drawdown     : {s['avg_max_drawdown']*100:.2f}%")
    print(f"  Worst drawdown   : {s['worst_drawdown']*100:.2f}%")
    print(f"  Avg period Sharpe: {s['avg_period_sharpe']:.2f}")
    print()

    for p in result['periods']:
        flag = '+' if p['outperformance'] > 0 else '-'
        print(
            f"    [{flag}] {p['period_start']} -> {p['period_end']}  "
            f"opt:{p['optimized_return']*100:+.1f}%  "
            f"eq:{p['equal_weight_return']*100:+.1f}%  "
            f"out:{p['outperformance']*100:+.1f}%  "
            f"mdd:{p['period_max_drawdown']*100:.1f}%  "
            f"top:{p['top_weight']}"
        )
    print()


# ── 7. Realized vs predicted ──────────────────────────────────────────────────

section("7 / REALIZED vs PREDICTED (MODERATE, FULL UNIVERSE)")

opt      = optimize_portfolio(investable, prices, risk='moderate')
realized = compute_portfolio_metrics(prices, opt['weights'])

ret_gap = opt['expected_return'] - realized['realized_annual_return']
vol_gap = opt['volatility']      - realized['realized_volatility']

print(f"  {'Metric':<22} {'Predicted':>12} {'Realized':>12} {'Gap':>10}")
print(f"  {'-'*58}")
print(f"  {'Annual Return':<22} {opt['expected_return']*100:>11.2f}% {realized['realized_annual_return']*100:>11.2f}% {ret_gap*100:>+9.2f}%")
print(f"  {'Volatility':<22} {opt['volatility']*100:>11.2f}% {realized['realized_volatility']*100:>11.2f}% {vol_gap*100:>+9.2f}%")
print(f"  {'Sharpe Ratio':<22} {opt['sharpe_ratio']:>12.2f} {realized['realized_sharpe']:>12.2f} {opt['sharpe_ratio']-realized['realized_sharpe']:>+10.2f}")
print(f"  {'Max Drawdown':<22} {'—':>12} {realized['max_drawdown']*100:>11.2f}% {'':>10}")
print(f"  {'Calmar Ratio':<22} {'—':>12} {realized['calmar_ratio']:>12.2f} {'':>10}")
print(f"  {'Total Return (5Y)':<22} {'—':>12} {realized['total_return']*100:>11.2f}% {'':>10}")

print(f"\n  Portfolio positions ({opt['n_positions']} holdings):")
for ticker, d in sorted(opt['allocation'].items(), key=lambda x: -x[1]['allocation']):
    label = clustered.loc[ticker, 'cluster_label'] if ticker in clustered.index else '?'
    print(f"    {ticker:6s} {d['shares']:3d} shares @ ${d['price']:7.2f} = ${d['allocation']:7,.0f}  ({d['weight']:.1%})  [{label}]")

if ret_gap > 0.15:
    warn(f"Return gap ({ret_gap*100:.1f}%) is high — optimizer may be overfitting to training data")
elif ret_gap > 0.05:
    info(f"Return gap ({ret_gap*100:.1f}%) is within normal in-sample bounds (<15%)")
else:
    ok(f"Return gap ({ret_gap*100:.1f}%) is low — excellent prediction accuracy")

if abs(vol_gap) < 0.02:
    ok(f"Volatility prediction excellent (gap: {vol_gap*100:.2f}%)")
else:
    warn(f"Volatility gap {vol_gap*100:.2f}% — covariance estimate may be off")


# ── 8. Scorecard ──────────────────────────────────────────────────────────────

section("8 / EVALUATION SCORECARD")

scores = {}

# Data quality
scores['Data completeness'] = 'PASS' if nan_pct < 20 else 'WARN'
scores['Ticker coverage']   = 'PASS' if not missing_prices else 'WARN'

# Feature quality
scores['Feature scaling']   = 'PASS' if nan_after_scale == 0 else 'FAIL'
scores['Cluster quality']   = 'PASS' if len(singleton_clusters) <= 4 else 'WARN'

# Backtest
if 'moderate' in backtest_results:
    mod = backtest_results['moderate']['summary']
    scores['Moderate win rate']    = 'PASS' if mod['win_rate_vs_equal'] >= 0.5  else 'WARN'
    scores['Moderate drawdown']    = 'PASS' if mod['worst_drawdown'] > -0.20    else 'WARN'
if 'aggressive' in backtest_results:
    agg = backtest_results['aggressive']['summary']
    scores['Aggressive drawdown']  = 'PASS' if agg['worst_drawdown'] > -0.30    else 'WARN'

# Prediction accuracy
scores['Return gap < 15%']  = 'PASS' if ret_gap < 0.15 else 'WARN'
scores['Vol prediction']    = 'PASS' if abs(vol_gap) < 0.02 else 'WARN'
scores['Realized Sharpe']   = 'PASS' if realized['realized_sharpe'] > 0.5 else 'WARN'

print()
for metric, result in scores.items():
    icon = '[OK]  ' if result == 'PASS' else ('[WARN]' if result == 'WARN' else '[FAIL]')
    print(f"  {icon}  {metric:<30} {result}")

passed = sum(1 for v in scores.values() if v == 'PASS')
warned = sum(1 for v in scores.values() if v == 'WARN')
failed = sum(1 for v in scores.values() if v == 'FAIL')
total  = len(scores)

print(f"\n  Score: {passed}/{total} PASS  |  {warned} WARN  |  {failed} FAIL")
print(f"\n  Total evaluation time: {time.time()-t0:.1f}s")
print(f"\n{SEP}\n")