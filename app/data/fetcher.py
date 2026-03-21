"""
Data Fetcher
------------
- fetch_prices()       : bulk price download via yf.download (unchanged)
- fetch_fundamentals() : parallel per-ticker fetch via ThreadPoolExecutor(20)
                         with tenacity retry + disk cache + PIT fundamentals

Speed optimisations vs v1:
  1. MAX_WORKERS 15 -> 20
  2. Prices fetched once in bulk via yf.download() before workers start
     (eliminates one t.history() HTTP call per ticker)
  3. Per-worker random jitter to avoid synchronised request bursts
  4. Workers skip .info call for tickers already in disk cache
"""

from __future__ import annotations

import time
import random
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from app.core.logger import get_logger
from app.core.disk_cache import DiskCache
from app.data.pit_fundamentals import calculate_pit_fundamentals

log           = get_logger(__name__)
_tenacity_log = logging.getLogger("tenacity")

MAX_WORKERS     = 20       # increased from 15
CACHE_TTL_HOURS = 24
JITTER_MIN_S    = 0.0      # min random sleep per worker
JITTER_MAX_S    = 0.5      # max random sleep per worker

_disk_cache = DiskCache(cache_dir="app/data/cache", ttl_hours=CACHE_TTL_HOURS)


def _make_retry():
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(_tenacity_log, logging.WARNING),
    )


@_make_retry()
def _fetch_single_ticker(
    ticker: str,
    cutoff_date: Optional[datetime],
    price_on_date: float,          # pre-fetched via bulk yf.download
) -> dict:
    """
    Fetch fundamental data for one ticker.
    Price is passed in from the bulk fetch — no t.history() call needed.
    """
    # Random jitter to avoid all workers hitting Yahoo simultaneously
    time.sleep(random.uniform(JITTER_MIN_S, JITTER_MAX_S))

    t = yf.Ticker(ticker)

    # Quarterly + annual data (5 calls -> 4 calls, history eliminated)
    q_financials       = t.quarterly_financials
    q_balance_sheet    = t.quarterly_balance_sheet
    q_income_stmt      = t.quarterly_income_stmt
    annual_income_stmt = t.income_stmt

    # .info last — slowest call, kept at end so other data is fetched first
    info               = t.info
    shares_outstanding = info.get("sharesOutstanding", np.nan)

    pit = calculate_pit_fundamentals(
        ticker=ticker,
        quarterly_financials=q_financials,
        quarterly_balance_sheet=q_balance_sheet,
        quarterly_income_stmt=q_income_stmt,
        annual_income_stmt=annual_income_stmt,
        price_on_date=price_on_date,
        shares_outstanding=shares_outstanding,
        cutoff_date=cutoff_date or datetime.today(),
    )

    pit.update({
        "sector":         info.get("sector",         "Unknown"),
        "pb_ratio":       info.get("priceToBook",    np.nan),
        "roe":            info.get("returnOnEquity", np.nan),
        "dividend_yield": info.get("dividendYield",  np.nan),
        "beta":           info.get("beta",           np.nan),
        "market_cap":     info.get("marketCap",      np.nan),
    })

    return pit


def _worker(
    ticker: str,
    cutoff_date: Optional[datetime],
    price_on_date: float,
) -> tuple[str, dict | None]:
    cache_key = f"{ticker}_{(cutoff_date or datetime.today()).strftime('%Y%m%d')}"

    cached = _disk_cache.get(cache_key)
    if cached is not None:
        log.info(f"cache_hit  {ticker}")
        return ticker, cached

    try:
        result = _fetch_single_ticker(ticker, cutoff_date, price_on_date)
        _disk_cache.set(cache_key, result)
        log.info(f"fetched    OK {ticker}")
        return ticker, result
    except Exception as e:
        log.warning(f"failed     ✗ {ticker}: {e}")
        return ticker, None


def _bulk_fetch_prices(
    tickers: list[str],
    cutoff_date: Optional[datetime],
) -> dict[str, float]:
    """
    Fetch closing prices for all tickers in a single yf.download() call.
    Returns dict mapping ticker -> price on cutoff date.
    Much faster than one t.history() call per ticker in each worker.
    """
    cutoff     = cutoff_date or datetime.today()
    end_date   = (cutoff + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (cutoff - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            return {t: np.nan for t in tickers}

        close = data["Close"] if "Close" in data else data
        prices = {}
        for ticker in tickers:
            try:
                col = ticker if ticker in close.columns else None
                if col:
                    series = close[col].dropna()
                    prices[ticker] = float(series.iloc[-1]) if not series.empty else np.nan
                else:
                    prices[ticker] = np.nan
            except Exception:
                prices[ticker] = np.nan
        return prices

    except Exception as e:
        log.warning(f"Bulk price fetch failed: {e}")
        return {t: np.nan for t in tickers}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Bulk price download — unchanged from original."""
    log.info(f"Fetching prices for {len(tickers)} tickers")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    return data["Close"]


def fetch_fundamentals(
    tickers: list[str],
    cutoff_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch PIT fundamentals for all tickers in parallel.

    Optimisations:
      - Prices fetched once in bulk before workers start
      - 20 workers (up from 15)
      - Per-worker jitter to avoid Yahoo rate limiting
      - Disk cache checked first per ticker

    Args:
        tickers:     list of stock symbols (up to 50 recommended)
        cutoff_date: point-in-time date (defaults to today)

    Returns:
        pd.DataFrame indexed by ticker
    """
    log.info(
        f"Fetching fundamentals for {len(tickers)} tickers "
        f"(workers={MAX_WORKERS}, cutoff={'today' if cutoff_date is None else cutoff_date})"
    )

    # Check which tickers need fetching (not in cache)
    today_str  = (cutoff_date or datetime.today()).strftime("%Y%m%d")
    uncached   = [
        t for t in tickers
        if _disk_cache.get(f"{t}_{today_str}") is None
    ]
    log.info(f"Cache: {len(tickers) - len(uncached)} hits, {len(uncached)} misses")

    # Bulk price fetch only for uncached tickers — single HTTP call
    prices_map: dict[str, float] = {}
    if uncached:
        log.info(f"Bulk fetching prices for {len(uncached)} uncached tickers")
        prices_map = _bulk_fetch_prices(uncached, cutoff_date)

    # Parallel fundamental fetch
    results: dict[str, dict] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ticker in tickers:
            price = prices_map.get(ticker, np.nan)
            future = executor.submit(_worker, ticker, cutoff_date, price)
            futures[future] = ticker

        for future in as_completed(futures):
            ticker, data = future.result()
            if data is not None:
                results[ticker] = data

    if not results:
        log.warning("No fundamentals fetched — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(results).T
    df.index.name = "ticker"

    numeric_cols = [
        "pe_ratio", "pb_ratio", "roe", "debt_to_equity",
        "revenue_growth", "dividend_yield", "beta", "market_cap", "eps_ttm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Fundamentals fetched: {len(df)}/{len(tickers)} tickers succeeded")
    return df