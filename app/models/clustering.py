import pandas as pd
from sklearn.cluster import KMeans
from app.core.logger import get_logger

log = get_logger(__name__)

CLUSTER_LABELS = {
    0: 'Growth',
    1: 'Value', 
    2: 'Defensive',
    3: 'Income',
    4: 'Momentum'
}

def cluster_stocks(scaled_df: pd.DataFrame, 
                   combined_df: pd.DataFrame,
                   n_clusters: int = 5) -> pd.DataFrame:
    log.info(f"Clustering into {n_clusters} groups")
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init = 'k-means++')
    combined = combined_df.copy()
    combined['cluster']       = km.fit_predict(scaled_df)
    combined['cluster_label'] = combined['cluster'].map(CLUSTER_LABELS)
    return combined