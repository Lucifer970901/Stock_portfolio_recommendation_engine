"""
Stock Clustering
----------------
Groups stocks into behaviorally similar clusters using KMeans on a
weighted scaled feature matrix. Labels are assigned per-ticker from
raw fundamental values — NOT from cluster centroids — making them
deterministic and always correct regardless of KMeans groupings.

Feature weights for KMeans:
  - Fundamental cols : 2.0x
  - Technical cols   : 1.0x
  - Engineered cols  : 0.5x

Optimal n_clusters=8 from elbow analysis on 50 tickers.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from app.core.logger import get_logger
from app.features.fundamentals import FUNDAMENTAL_COLS, TECHNICAL_COLS, ENGINEERED_COLS

log = get_logger(__name__)

DEFAULT_N_CLUSTERS = 8

FEATURE_WEIGHTS = (
    {col: 2.0 for col in FUNDAMENTAL_COLS} |
    {col: 1.0 for col in TECHNICAL_COLS}   |
    {col: 0.5 for col in ENGINEERED_COLS}
)


def _apply_weights(scaled_df: pd.DataFrame) -> pd.DataFrame:
    weights   = pd.Series(FEATURE_WEIGHTS)
    available = weights.index.intersection(scaled_df.columns)
    weighted  = scaled_df.copy()
    weighted[available] = scaled_df[available] * weights[available]
    return weighted


def _ticker_label(row: pd.Series) -> str:
    """
    Assign a human-readable label to a single ticker based on its
    raw fundamental values. Applied per-row, fully deterministic.

    Priority order — most distinctive conditions checked first.
    """
    eps        = row.get('eps_ttm',        np.nan)
    de         = row.get('debt_to_equity', np.nan)
    rev_growth = row.get('revenue_growth', np.nan)
    pe         = row.get('pe_ratio',       np.nan)
    beta       = row.get('beta',           np.nan)
    roe        = row.get('roe',            np.nan)
    div        = row.get('dividend_yield', np.nan)
    vol        = row.get('volatility',     np.nan)
    sector     = row.get('sector',         '')

    # Loss-making
    if pd.notna(eps) and eps <= 0:
        return 'Distressed'

    # Negative equity
    if pd.notna(de) and de < -3:
        return 'Negative Equity'

    # Hypergrowth: >50% revenue growth
    if pd.notna(rev_growth) and rev_growth > 0.50:
        return 'Hypergrowth'

    # Speculative: very high PE + high beta
    if pd.notna(pe) and pd.notna(beta) and pe > 150 and beta > 1.5:
        return 'Speculative'

    # Quality Growth: high ROE + solid revenue growth
    if pd.notna(roe) and pd.notna(rev_growth) and roe > 0.25 and rev_growth > 0.10:
        return 'Quality Growth'

    # Defensive Income: low beta + dividend payer
    if pd.notna(beta) and pd.notna(div) and beta < 0.7 and div > 0.015:
        return 'Defensive Income'

    # Sector-based fallbacks
    sector_map = {
        'Energy':                 'Energy',
        'Healthcare':             'Healthcare',
        'Financial Services':     'Financials',
        'Technology':             'Technology',
        'Communication Services': 'Technology',
        'Consumer Defensive':     'Consumer Defensive',
        'Consumer Cyclical':      'Consumer Cyclical',
        'Industrials':            'Industrials',
        'Utilities':              'Utilities',
        'Real Estate':            'Real Estate',
    }
    if sector in sector_map:
        return sector_map[sector]

    return 'Blend'


def cluster_stocks(
    scaled_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> pd.DataFrame:
    """
    Cluster stocks using KMeans on weighted features, label per-ticker.

    Args:
        scaled_df:   scaled feature DataFrame from scale_features()
        combined_df: original merged DataFrame with raw fundamentals + 'sector'
        n_clusters:  number of clusters (default=8 from elbow analysis)

    Returns:
        combined_df with 'cluster' (int) and 'cluster_label' (str) columns added
    """
    log.info(f"Clustering {len(scaled_df)} tickers into {n_clusters} groups")

    weighted_df = _apply_weights(scaled_df)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        init='k-means++',
    )
    cluster_ids = km.fit_predict(weighted_df)

    result             = combined_df.copy()
    result['cluster']  = cluster_ids

    # Assign labels per-ticker from raw fundamentals — deterministic
    result['cluster_label'] = result.apply(_ticker_label, axis=1)

    # Log cluster label summary
    summary = (
        result.groupby('cluster_label')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    log.info(f"Cluster summary:\n{summary.to_string(index=False)}")

    # Size warnings based on KMeans clusters (not labels)
    km_counts = pd.Series(cluster_ids).value_counts()
    if km_counts.min() < 3:
        log.warning(
            f"Smallest KMeans cluster has {km_counts.min()} ticker(s) — "
            f"consider reducing n_clusters"
        )
    if km_counts.max() > n_clusters * 3:
        log.warning(
            f"Largest KMeans cluster has {km_counts.max()} tickers — "
            f"clusters may be unbalanced"
        )

    return result


def get_cluster_stats(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-label mean statistics sorted by PE ratio."""
    stat_cols = [
        col for col in [
            'pe_ratio', 'revenue_growth', 'debt_to_equity',
            'roe', 'beta', 'dividend_yield', 'market_cap',
            'volatility', 'momentum_3m',
        ]
        if col in clustered_df.columns
    ]
    return (
        clustered_df
        .groupby('cluster_label')[stat_cols]
        .mean()
        .round(3)
        .sort_values('pe_ratio', ascending=True)
    )