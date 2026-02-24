import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from app.core.logger import get_logger

log = get_logger(__name__)

FEATURE_COLS = [
    'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity',
    'revenue_growth', 'dividend_yield', 'beta', 'market_cap',
    'momentum_3m', 'momentum_6m', 'volatility', 'rsi'
]

def merge_features(fundamentals: pd.DataFrame, 
                   technical: pd.DataFrame) -> pd.DataFrame:
    return fundamentals.join(technical, how='inner')

def scale_features(combined: pd.DataFrame):
    df = combined[FEATURE_COLS].copy()
    
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    
    imputed = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index, columns=FEATURE_COLS
    )
    scaled = pd.DataFrame(
        scaler.fit_transform(imputed),
        index=df.index, columns=FEATURE_COLS
    )
    
    log.info(f"Scaled features shape: {scaled.shape}")
    return scaled, scaler, imputer