import time
import pandas as pd
from app.core.config import settings
from app.core.logger import get_logger
from app.data.fetcher import fetch_prices, fetch_fundamentals
from app.features.technical import compute_technical_features
from app.features.fundamentals import merge_features, scale_features
from app.models.similarity import (
    build_similarity_matrices,
    get_similar_stocks,
    get_complementary_stocks,
)
from app.models.clustering import cluster_stocks
from app.models.optimizer import optimize_portfolio
from app.core.cache import cache

log = get_logger(__name__)

# Cluster labels excluded from optimization universe
# These are behaviorally unsuitable for portfolio construction
EXCLUDE_FROM_OPTIMIZATION = {'Distressed', 'Negative Equity'}


class RecommenderService:
    def __init__(self):
        self.prices            = None
        self.combined_df       = None
        self.scaled_df         = None
        self.similarity_df     = None   # combined similarity (backward compat)
        self._similarity_mats  = None   # full dict: fundamental/technical/combined
        self.investable_tickers: list[str] = []
        self.is_ready          = False
        self.built_at          = None

    def build(self, tickers: list[str] = None):
        tickers = tickers or settings.tickers
        log.info("Building recommender...")

        self.prices          = fetch_prices(tickers)
        fundamentals         = fetch_fundamentals(tickers)
        technical            = compute_technical_features(self.prices)
        combined             = merge_features(fundamentals, technical)
        self.scaled_df, _, _ = scale_features(combined)
        self.combined_df     = cluster_stocks(self.scaled_df, combined)

        # Build all three similarity matrices
        self._similarity_mats = build_similarity_matrices(self.scaled_df)
        self.similarity_df    = self._similarity_mats['combined']  # backward compat

        # Investable universe — exclude distressed / negative equity clusters
        self.investable_tickers = self._build_investable_universe()
        
        self.is_ready = True
        self.built_at = time.time()
        cache.invalidate()
        log.info(
            f"Recommender ready — "
            f"universe: {len(self.combined_df)}, "
            f"investable: {len(self.investable_tickers)}"
        )

    def _build_investable_universe(self) -> list[str]:
        """
        Filter out tickers unsuitable for portfolio optimization.

        Excludes based on fundamentals directly (not cluster labels):
          - Loss-making: negative or missing EPS (eps_ttm <= 0)
          - Negative equity: debt_to_equity < -3
          - Insufficient price history: < 60 trading days

        Returns:
            list of investable ticker symbols
        """
        excluded: dict[str, str] = {}

        for ticker in self.combined_df.index:
            # Check price history
            if (ticker not in self.prices.columns or
                    self.prices[ticker].dropna().shape[0] < 60):
                excluded[ticker] = 'insufficient_history'
                continue

            row = self.combined_df.loc[ticker]

            # Loss-making: eps_ttm <= 0 or missing
            eps = row.get('eps_ttm', None)
            if eps is not None and pd.notna(eps) and eps <= 0:
                excluded[ticker] = f'loss_making (eps_ttm={eps:.2f})'
                continue

            # Negative equity: extreme negative D/E
            de = row.get('debt_to_equity', None)
            if de is not None and pd.notna(de) and de < -3:
                excluded[ticker] = f'negative_equity (d/e={de:.2f})'
                continue
            # Add after the negative equity check
            pe   = row.get('pe_ratio',  None)
            beta = row.get('beta',      None)
            if (pd.notna(pe) and pe > 300 and
                pd.notna(beta) and beta > 1.5):
                excluded[ticker] = f'speculative (pe={pe:.0f}, beta={beta:.2f})'
            continue        
        investable = [t for t in self.combined_df.index if t not in excluded]

        if excluded:
            for ticker, reason in excluded.items():
                log.info(f"Excluded from optimization: {ticker} — {reason}")

        return investable

    def similar(self, ticker: str, top_n: int = 5) -> list[dict]:
        self._check_ready()

        key    = f"similar:{ticker}:{top_n}"
        cached = cache.get(key)
        if cached:
            return cached

        result = get_similar_stocks(
            ticker, self.similarity_df, self.combined_df, top_n
        )
        result = result.reset_index().to_dict(orient='records')
        cache.set(key, result)
        return result

    def complementary(self, ticker: str, top_n: int = 5) -> list[dict]:
        """Find most diversifying stocks for a given ticker."""
        self._check_ready()

        key    = f"complementary:{ticker}:{top_n}"
        cached = cache.get(key)
        if cached:
            return cached

        result = get_complementary_stocks(
            ticker, self.similarity_df, self.combined_df, top_n
        )
        result = result.reset_index().to_dict(orient='records')
        cache.set(key, result)
        return result

    def gaps(self, portfolio: list[str], top_n: int = 5) -> list[dict]:
        self._check_ready()

        key    = f"gaps:{':'.join(sorted(portfolio))}:{top_n}"
        cached = cache.get(key)
        if cached:
            return cached

        port_returns = (
            self.prices[portfolio].pct_change().dropna().mean(axis=1)
        )

        correlations = {}
        for ticker in self.combined_df.index:
            if ticker in portfolio:
                continue
            corr = self.prices[ticker].pct_change().dropna().corr(port_returns)
            correlations[ticker] = corr

        rec_df = (
            pd.Series(correlations)
            .sort_values()
            .head(top_n)
            .reset_index()
        )
        rec_df.columns = ['ticker', 'correlation']

        rec_df['sector'] = rec_df['ticker'].map(
            self.combined_df['sector'].to_dict()
        )

        result = rec_df.to_dict(orient='records')
        cache.set(key, result)
        return result

    def optimize(self, tickers: list[str], risk: str = 'moderate') -> dict:
        self._check_ready()

        # Filter requested tickers to investable universe only
        investable = [t for t in tickers if t in self.investable_tickers]
        excluded   = [t for t in tickers if t not in self.investable_tickers]

        if excluded:
            log.warning(
                f"Excluded from optimization (not investable): {excluded}"
            )

        if len(investable) < 2:
            raise ValueError(
                f"Need at least 2 investable tickers — "
                f"got {len(investable)} after filtering. "
                f"Excluded: {excluded}"
            )

        key    = f"optimize:{':'.join(sorted(investable))}:{risk}"
        cached = cache.get(key)
        if cached:
            return cached

        result = optimize_portfolio(investable, self.prices, risk)
        cache.set(key, result)
        return result

    def _check_ready(self):
        if not self.is_ready:
            raise RuntimeError("Call .build() first")


# Singleton
recommender = RecommenderService()