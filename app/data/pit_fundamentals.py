"""
Point-in-Time (PIT) Fundamentals Calculator

Calculates fundamental metrics from quarterly reports using only data
that would have been publicly available on or before a given cutoff date.

Reporting lag: 45 days after quarter end (conservative filing assumption).
Revenue growth uses annual income statement for clean YoY comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from app.core.logger import get_logger

log = get_logger(__name__)

REPORTING_LAG_DAYS = 45


def _available_quarters(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return only quarters publicly available by cutoff date."""
    if df is None or df.empty:
        return pd.DataFrame()
    available_cols = [
        col for col in df.columns
        if pd.Timestamp(col) + timedelta(days=REPORTING_LAG_DAYS) <= cutoff
    ]
    if not available_cols:
        return pd.DataFrame()
    return df[sorted(available_cols, reverse=True)]


def _available_annual(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Return only fiscal years publicly available by cutoff date.
    Annual reports are assumed filed ~90 days after fiscal year end.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    available_cols = [
        col for col in df.columns
        if pd.Timestamp(col) + timedelta(days=90) <= cutoff
    ]
    if not available_cols:
        return pd.DataFrame()
    return df[sorted(available_cols, reverse=True)]


def _safe_get(df: pd.DataFrame, row_key: str, col_index: int = 0) -> float:
    """Safely extract a single scalar from a quarterly DataFrame."""
    try:
        if row_key in df.index and col_index < len(df.columns):
            val = df.iloc[df.index.get_loc(row_key), col_index]
            return float(val) if pd.notna(val) else np.nan
    except Exception:
        pass
    return np.nan


def _ttm_sum(df: pd.DataFrame, row_key: str, n_quarters: int = 4) -> float:
    """Sum up to n_quarters values for a row (Trailing Twelve Months)."""
    try:
        if row_key not in df.index:
            return np.nan
        vals  = df.loc[row_key].iloc[:n_quarters]
        valid = vals.dropna()
        if len(valid) < 2:
            return np.nan
        return float(valid.sum())
    except Exception:
        return np.nan


def _yoy_growth(row: pd.Series) -> float:
    """
    Calculate clean year-over-year growth from an annual Series.
    Uses the two most recent available years: (year0 - year1) / year1
    """
    try:
        valid = row.dropna()
        if len(valid) < 2:
            return np.nan
        year0 = float(valid.iloc[0])  # most recent year
        year1 = float(valid.iloc[1])  # prior year
        if year1 == 0:
            return np.nan
        return round((year0 - year1) / abs(year1), 4)
    except Exception:
        return np.nan


def calculate_pit_fundamentals(
    ticker: str,
    quarterly_financials: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    quarterly_income_stmt: pd.DataFrame,
    annual_income_stmt: pd.DataFrame,
    price_on_date: float,
    shares_outstanding: float = np.nan,
    cutoff_date: Optional[datetime] = None,
) -> dict:
    """
    Calculate point-in-time fundamental metrics for a single ticker.

    Args:
        ticker:                  stock symbol
        quarterly_financials:    yf.Ticker.quarterly_financials
        quarterly_balance_sheet: yf.Ticker.quarterly_balance_sheet
        quarterly_income_stmt:   yf.Ticker.quarterly_income_stmt
        annual_income_stmt:      yf.Ticker.income_stmt (for clean YoY revenue)
        price_on_date:           closing price on the cutoff date
        shares_outstanding:      from ticker.info['sharesOutstanding']
        cutoff_date:             as-of date (defaults to today)

    Returns:
        dict with PIT fundamental metrics
    """
    cutoff = pd.Timestamp(cutoff_date or datetime.today())

    result = {
        "ticker":         ticker,
        "as_of_date":     cutoff.date().isoformat(),
        "pe_ratio":       np.nan,
        "eps_ttm":        np.nan,
        "revenue_growth": np.nan,
        "debt_to_equity": np.nan,
        "pb_ratio":       np.nan,
        "roe":            np.nan,
        "dividend_yield": np.nan,
        "beta":           np.nan,
        "market_cap":     np.nan,
        "sector":         "Unknown",
    }

    # EPS TTM = Net Income TTM / shares outstanding
    try:
        if quarterly_income_stmt is not None and not quarterly_income_stmt.empty:
            avail = _available_quarters(quarterly_income_stmt, cutoff)
            if not avail.empty:
                ni_key = next(
                    (k for k in ["Net Income", "NetIncome"] if k in avail.index),
                    None
                )
                if ni_key:
                    ttm_ni = _ttm_sum(avail, ni_key, 4)
                    if pd.notna(ttm_ni) and pd.notna(shares_outstanding) and shares_outstanding > 0:
                        result["eps_ttm"] = round(ttm_ni / shares_outstanding, 4)
                    else:
                        result["eps_ttm"] = ttm_ni
    except Exception as e:
        log.warning(f"EPS TTM failed for {ticker}", error=str(e))

    # P/E = price / EPS TTM
    if pd.notna(result["eps_ttm"]) and result["eps_ttm"] > 0 and pd.notna(price_on_date):
        result["pe_ratio"] = round(price_on_date / result["eps_ttm"], 2)

    # Revenue Growth YoY — uses annual income statement for clean comparison
    # (year0 - year1) / year1 using the two most recent available fiscal years
    try:
        if annual_income_stmt is not None and not annual_income_stmt.empty:
            avail_annual = _available_annual(annual_income_stmt, cutoff)
            if not avail_annual.empty:
                rev_key = next(
                    (k for k in ["Total Revenue", "Revenue", "TotalRevenue"]
                     if k in avail_annual.index),
                    None
                )
                if rev_key:
                    growth = _yoy_growth(avail_annual.loc[rev_key])
                    if pd.notna(growth):
                        result["revenue_growth"] = growth
    except Exception as e:
        log.warning(f"Revenue growth failed for {ticker}", error=str(e))

    # Debt / Equity = Total Debt / Stockholders Equity (latest available quarter)
    try:
        if quarterly_balance_sheet is not None and not quarterly_balance_sheet.empty:
            avail_bs = _available_quarters(quarterly_balance_sheet, cutoff)
            if not avail_bs.empty:
                debt_key = next(
                    (k for k in ["Total Debt", "Long Term Debt", "TotalDebt"]
                     if k in avail_bs.index),
                    None
                )
                equity_key = next(
                    (k for k in [
                        "Stockholders Equity",
                        "Total Stockholder Equity",
                        "StockholdersEquity",
                        "Common Stock Equity",
                    ] if k in avail_bs.index),
                    None
                )
                if debt_key and equity_key:
                    debt   = _safe_get(avail_bs, debt_key)
                    equity = _safe_get(avail_bs, equity_key)
                    if pd.notna(debt) and pd.notna(equity) and equity != 0:
                        result["debt_to_equity"] = round(debt / equity, 4)
    except Exception as e:
        log.warning(f"Debt/Equity failed for {ticker}", error=str(e))

    return result