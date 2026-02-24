import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from app.core.logger import get_logger

log = get_logger(__name__)

def build_similarity_matrix(scaled_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building similarity matrix")
    matrix = cosine_similarity(scaled_df)
    return pd.DataFrame(matrix, index=scaled_df.index, columns=scaled_df.index)

def get_similar_stocks(ticker: str, 
                        similarity_df: pd.DataFrame,
                        combined_df: pd.DataFrame,
                        top_n: int = 5) -> pd.DataFrame:
    if ticker not in similarity_df.index:
        raise ValueError(f"{ticker} not in universe")
    
    similar = similarity_df[ticker].drop(ticker).nlargest(top_n)
    result  = combined_df.loc[similar.index, 
                               ['sector', 'beta', 'momentum_6m', 'volatility']]
    result['similarity'] = similar.round(3)
    return result