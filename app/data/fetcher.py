import yfinance as yf
import pandas as pd
import numpy as np
from app.core.logger import get_logger

log = get_logger(__name__)

def fetch_prices(tickers: list[str], period: str = '2y') -> pd.DataFrame:
    log.info(f"Fetching prices for {len(tickers)} tickers")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    return data['Close']

def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    log.info("Fetching fundamentals")
    records = []
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append({
                'ticker':         ticker,
                'sector':         info.get('sector', 'Unknown'),
                'pe_ratio':       info.get('trailingPE', np.nan),
                'pb_ratio':       info.get('priceToBook', np.nan),
                'roe':            info.get('returnOnEquity', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'dividend_yield': info.get('dividendYield', np.nan),
                'beta':           info.get('beta', np.nan),
                'market_cap':     info.get('marketCap', np.nan),
            })
            log.info(f"  ✓ {ticker}")
        except Exception as e:
            log.warning(f"  ✗ {ticker}: {e}")
    
    if not records:                                  
        return pd.DataFrame()
    
    return pd.DataFrame(records).set_index('ticker')