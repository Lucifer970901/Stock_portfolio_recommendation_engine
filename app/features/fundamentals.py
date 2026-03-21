import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from app.core.logger import get_logger

log = get_logger(__name__)

# ── Base feature columns from fundamentals + technical pipeline ───────────────
FUNDAMENTAL_COLS = [
    'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity',
    'revenue_growth', 'beta', 'market_cap_log',
]

TECHNICAL_COLS = [
    'momentum_3m', 'momentum_6m', 'volatility', 'rsi',
]

# Engineered binary/derived columns added during preprocessing
ENGINEERED_COLS = [
    'is_profitable',          # 1 if pe_ratio > 0 else 0
    'has_dividend',           # 1 if dividend_yield > 0 else 0
    'dividend_yield_amount',  # actual yield (0 if no dividend)
    'has_momentum_history',   # 1 if momentum_3m and momentum_6m are both available
]

# Final feature set passed to scaler — order matters for inverse_transform
FEATURE_COLS = FUNDAMENTAL_COLS + TECHNICAL_COLS + ENGINEERED_COLS

# Clipping bounds applied before scaling (after engineered features are created)
# NaN values are preserved through clipping
CLIP_BOUNDS = {
    'pe_ratio':       (0.0,   200.0),
    'pb_ratio':       (0.0,   50.0),
    'roe':            (-2.0,  3.0),    # keep negative ROE, clip only extremes
    'debt_to_equity': (-5.0,  10.0),   # negative equity companies clipped
    'revenue_growth': (-0.5,  2.0),
    'market_cap_log': (None,  None),   # no clip needed after log transform
}

# Columns that use median imputation (missing = data unavailable, not meaningful zero)
MEDIAN_IMPUTE_COLS = [
    'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity',
    'revenue_growth', 'beta',
    'momentum_3m', 'momentum_6m', 'volatility', 'rsi',
    'market_cap_log',
]

# Columns that use zero imputation (missing = zero is the correct value)
ZERO_IMPUTE_COLS = [
    'dividend_yield_amount',  # no dividend = 0 yield
]

# Binary columns — never need imputation (always 0 or 1)
BINARY_COLS = [
    'is_profitable',
    'has_dividend',
    'has_momentum_history',
]


def merge_features(
    fundamentals: pd.DataFrame,
    technical: pd.DataFrame,
) -> pd.DataFrame:
    """Join fundamental and technical feature DataFrames on ticker index."""
    merged = fundamentals.join(technical, how='inner')
    log.info(f"Merged features shape: {merged.shape}")
    return merged


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived and binary features from raw columns.
    Must run BEFORE clipping so original values drive the flags.
    """
    df = df.copy()

    # is_profitable: 1 if pe_ratio is positive (company is earning), else 0
    # NaN pe_ratio (e.g. INTC with negative EPS) -> 0 (treat as not profitable)
    df['is_profitable'] = (
        df['pe_ratio'].fillna(0).gt(0)
    ).astype(int)

    # has_dividend / dividend_yield_amount
    # NaN dividend_yield means no dividend data -> treat as 0
    div = df['dividend_yield'].fillna(0)
    df['has_dividend']          = div.gt(0).astype(int)
    df['dividend_yield_amount'] = div.clip(lower=0)  # ensure non-negative

    # market_cap_log: log1p transform to compress NVDA/AAPL vs smaller caps
    # NaN market_cap stays NaN — imputed later with median
    df['market_cap_log'] = np.log1p(df['market_cap'].clip(lower=0))

    # has_momentum_history: 1 if both momentum cols are available
    df['has_momentum_history'] = (
        df['momentum_3m'].notna() & df['momentum_6m'].notna()
    ).astype(int)

    log.info(
        "Engineered features — "
        f"profitable: {df['is_profitable'].sum()}, "
        f"pays_dividend: {df['has_dividend'].sum()}, "
        f"has_momentum: {df['has_momentum_history'].sum()}"
    )

    return df


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip outlier-prone columns to sensible bounds, preserving NaN values."""
    df = df.copy()
    for col, (low, high) in CLIP_BOUNDS.items():
        if col not in df.columns or (low is None and high is None):
            continue
        n_clipped = ((df[col] < low) | (df[col] > high)).sum()
        df[col]   = df[col].clip(lower=low, upper=high)
        if n_clipped > 0:
            log.info(f"Clipped {n_clipped} values in '{col}' to [{low}, {high}]")
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column imputation strategy:
      - Binary cols    : fill with 0 (already set, just a safety net)
      - Zero cols      : fill with 0 (missing dividend = no dividend)
      - Median cols    : fill with column median
    """
    df = df.copy()

    # Binary — should already be 0/1, fill just in case
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Zero imputation
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Median imputation
    missing = {c: int(df[c].isna().sum()) for c in MEDIAN_IMPUTE_COLS if c in df.columns}
    missing = {k: v for k, v in missing.items() if v > 0}
    if missing:
        log.info(f"Median imputing columns: {missing}")

    median_imputer = SimpleImputer(strategy='median')
    median_cols    = [c for c in MEDIAN_IMPUTE_COLS if c in df.columns]
    df[median_cols] = median_imputer.fit_transform(df[median_cols])

    return df, median_imputer


def scale_features(combined: pd.DataFrame):
    """
    Full preprocessing pipeline: engineer -> clip -> impute -> scale.

    Args:
        combined: merged DataFrame from merge_features()

    Returns:
        scaled:          StandardScaler-transformed DataFrame (FEATURE_COLS)
        scaler:          fitted StandardScaler
        median_imputer:  fitted SimpleImputer for median cols
    """
    # Step 1: engineer derived + binary features
    df = _engineer_features(combined)

    # Step 2: clip outliers (after engineering so flags use raw values)
    df = _clip_outliers(df)

    # Step 3: select only final feature cols
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        log.warning(f"Missing expected feature columns: {missing_cols}")
    df = df[available_cols]

    # Step 4: per-column imputation
    df, median_imputer = _impute(df)

    # Step 5: standardise
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=available_cols,
    )

    log.info(f"Scaled features shape: {scaled.shape}")
    return scaled, scaler, median_imputer