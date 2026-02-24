import pandas as pd
import numpy as np
from app.core.logger import get_logger

log = get_logger(__name__)

def compute_rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss
    return (100 - 100 / (1 + rs)).iloc[-1]

def compute_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing technical features")
    records = {}
    
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        if len(close) < 60:
            continue
        returns = close.pct_change().dropna()
        records[ticker] = {
            'momentum_3m': close.pct_change(63).iloc[-1],
            'momentum_6m': close.pct_change(126).iloc[-1],
            'volatility':  returns.std() * np.sqrt(252),
            'rsi':         compute_rsi(close),
        }
    
    return pd.DataFrame(records).T