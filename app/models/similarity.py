"""
Stock Similarity
----------------
Builds similarity matrices from scaled features using cosine similarity.

Three matrices are computed:
  - fundamental_sim : based on valuation/growth features only (PE, ROE, etc.)
  - technical_sim   : based on momentum/volatility features only
  - combined_sim    : weighted blend (70% fundamental, 30% technical)

Using separate matrices preserves cross-sector behavioral insights:
  - AAPL and LLY may be fundamentally similar (quality growth) even across sectors
  - XOM and KO may be technically similar (low momentum, high dividend) even
    if their fundamentals differ

Two query functions:
  - get_similar_stocks()       : most similar tickers (for substitution)
  - get_complementary_stocks() : least similar tickers (for diversification)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.logger import get_logger
from app.features.fundamentals import FUNDAMENTAL_COLS, TECHNICAL_COLS

log = get_logger(__name__)

# Blend weights for combined similarity
FUNDAMENTAL_WEIGHT = 0.70
TECHNICAL_WEIGHT   = 0.30


def _cosine_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity matrix and return as labeled DataFrame."""
    matrix = cosine_similarity(df)
    return pd.DataFrame(matrix, index=df.index, columns=df.index)


def build_similarity_matrices(scaled_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build three similarity matrices from scaled feature DataFrame.

    Args:
        scaled_df: scaled feature DataFrame from scale_features()

    Returns:
        dict with keys:
          'fundamental' : similarity on valuation/growth features
          'technical'   : similarity on momentum/volatility features
          'combined'    : weighted blend (70% fundamental, 30% technical)
    """
    log.info(f"Building similarity matrices for {len(scaled_df)} tickers")

    fund_cols = [c for c in FUNDAMENTAL_COLS if c in scaled_df.columns]
    tech_cols = [c for c in TECHNICAL_COLS   if c in scaled_df.columns]

    fund_sim = _cosine_df(scaled_df[fund_cols]) if fund_cols else None
    tech_sim = _cosine_df(scaled_df[tech_cols]) if tech_cols else None

    if fund_sim is not None and tech_sim is not None:
        combined_sim = (
            FUNDAMENTAL_WEIGHT * fund_sim +
            TECHNICAL_WEIGHT   * tech_sim
        )
    elif fund_sim is not None:
        combined_sim = fund_sim
    else:
        combined_sim = tech_sim

    log.info(
        f"Similarity matrices built — "
        f"fundamental cols: {len(fund_cols)}, "
        f"technical cols: {len(tech_cols)}"
    )

    return {
        'fundamental': fund_sim,
        'technical':   tech_sim,
        'combined':    combined_sim,
    }


def build_similarity_matrix(scaled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper — returns combined similarity matrix.
    Use build_similarity_matrices() for full control.
    """
    return build_similarity_matrices(scaled_df)['combined']


def get_similar_stocks(
    ticker: str,
    similarity_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    top_n: int = 5,
    same_cluster: bool = False,
) -> pd.DataFrame:
    """
    Find the most similar stocks to a given ticker.
    Useful for finding substitutes within a portfolio.

    Args:
        ticker:        target ticker
        similarity_df: similarity matrix (fundamental, technical, or combined)
        combined_df:   merged DataFrame with fundamental + technical features
        top_n:         number of similar stocks to return
        same_cluster:  if True, restrict to same cluster_label as ticker

    Returns:
        DataFrame with similarity score and key metrics
    """
    if ticker not in similarity_df.index:
        raise ValueError(f"{ticker} not found in similarity matrix")

    candidates = similarity_df[ticker].drop(ticker)

    if same_cluster and 'cluster_label' in combined_df.columns:
        ticker_cluster = combined_df.loc[ticker, 'cluster_label']
        cluster_tickers = combined_df[
            combined_df['cluster_label'] == ticker_cluster
        ].index
        candidates = candidates[candidates.index.isin(cluster_tickers)]

    similar = candidates.nlargest(top_n)

    display_cols = [
        c for c in ['sector', 'cluster_label', 'pe_ratio', 'revenue_growth',
                    'beta', 'momentum_6m', 'volatility']
        if c in combined_df.columns
    ]
    result               = combined_df.loc[similar.index, display_cols].copy()
    result['similarity'] = similar.round(4)
    return result.sort_values('similarity', ascending=False)


def get_complementary_stocks(
    ticker: str,
    similarity_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    top_n: int = 5,
    exclude_same_cluster: bool = True,
) -> pd.DataFrame:
    """
    Find the least similar (most diversifying) stocks to a given ticker.
    Useful for building a diversified portfolio around a core holding.

    Args:
        ticker:               target ticker
        similarity_df:        similarity matrix
        combined_df:          merged DataFrame
        top_n:                number of complementary stocks to return
        exclude_same_cluster: if True, exclude stocks in same cluster

    Returns:
        DataFrame with similarity score and key metrics
    """
    if ticker not in similarity_df.index:
        raise ValueError(f"{ticker} not found in similarity matrix")

    candidates = similarity_df[ticker].drop(ticker)

    if exclude_same_cluster and 'cluster_label' in combined_df.columns:
        ticker_cluster  = combined_df.loc[ticker, 'cluster_label']
        same_cluster    = combined_df[
            combined_df['cluster_label'] == ticker_cluster
        ].index
        candidates = candidates[~candidates.index.isin(same_cluster)]

    complementary = candidates.nsmallest(top_n)

    display_cols = [
        c for c in ['sector', 'cluster_label', 'pe_ratio', 'revenue_growth',
                    'beta', 'momentum_6m', 'volatility']
        if c in combined_df.columns
    ]
    result               = combined_df.loc[complementary.index, display_cols].copy()
    result['similarity'] = complementary.round(4)
    return result.sort_values('similarity', ascending=True)


def similarity_report(
    ticker: str,
    scaled_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """
    Print a full similarity report for a ticker showing:
    - Fundamentally similar stocks
    - Technically similar stocks
    - Combined similar stocks
    - Complementary (diversifying) stocks

    Args:
        ticker:      target ticker
        scaled_df:   scaled feature DataFrame
        combined_df: merged DataFrame
        top_n:       number of results per section
    """
    matrices = build_similarity_matrices(scaled_df)

    print(f"\n{'='*55}")
    print(f"  Similarity Report: {ticker}")
    print(f"{'='*55}")

    print(f"\n[Fundamentally Similar] — valuation & growth profile")
    print(get_similar_stocks(ticker, matrices['fundamental'], combined_df, top_n).to_string())

    print(f"\n[Technically Similar] — momentum & volatility profile")
    print(get_similar_stocks(ticker, matrices['technical'], combined_df, top_n).to_string())

    print(f"\n[Combined Similar] — overall similarity")
    print(get_similar_stocks(ticker, matrices['combined'], combined_df, top_n).to_string())

    print(f"\n[Complementary] — most diversifying additions")
    print(get_complementary_stocks(ticker, matrices['combined'], combined_df, top_n).to_string())
    print()