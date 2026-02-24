import pandas as pd
from app.core.config import settings
from app.core.logger import get_logger
from app.data.fetcher import fetch_prices, fetch_fundamentals
from app.features.technical import compute_technical_features
from app.features.fundamental import merge_features, scale_features
from app.models.similarity import build_similarity_matrix, get_similar_stocks
from app.models.clustering import cluster_stocks
from app.models.optimizer import optimize_portfolio

log = get_logger(__name__)

class RecommenderService:
    def __init__(self):
        self.prices        = None
        self.combined_df   = None
        self.scaled_df     = None
        self.similarity_df = None
        self.is_ready      = False

    def build(self, tickers: list[str] = None):
        tickers = tickers or settings.tickers
        log.info("Building recommender...")
        
        self.prices      = fetch_prices(tickers)
        fundamentals     = fetch_fundamentals(tickers)
        technical        = compute_technical_features(self.prices)
        combined         = merge_features(fundamentals, technical)
        self.scaled_df, _, _ = scale_features(combined)
        self.combined_df = cluster_stocks(self.scaled_df, combined)
        self.similarity_df = build_similarity_matrix(self.scaled_df)
        
        self.is_ready = True
        log.info("Recommender ready")

    def similar(self, ticker: str, top_n: int = 5) -> dict:
        self._check_ready()
        result = get_similar_stocks(
            ticker, self.similarity_df, self.combined_df, top_n
        )
        return result.reset_index().to_dict(orient='records')

    def gaps(self, portfolio: list[str], top_n: int = 5) -> dict:
        self._check_ready()
        port_returns = (
            self.prices[portfolio].pct_change().dropna().mean(axis=1)
        )
        correlations = {}
        for ticker in self.combined_df.index:
            if ticker in portfolio:
                continue
            corr = self.prices[ticker].pct_change().dropna().corr(port_returns)
            correlations[ticker] = corr
        
        rec_df = (pd.Series(correlations)
                    .sort_values()
                    .head(top_n)
                    .reset_index())
        rec_df.columns = ['ticker', 'correlation']
        return rec_df.to_dict(orient='records')

    def optimize(self, tickers: list[str], 
                  risk: str = 'moderate') -> dict:
        self._check_ready()
        return optimize_portfolio(tickers, self.prices, risk)

    def _check_ready(self):
        if not self.is_ready:
            raise RuntimeError("Call .build() first")

# Singleton
recommender = RecommenderService()