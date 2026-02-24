from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    app_env:          str     = 'development'
    app_port:         int     = 8000
    log_level:        str     = 'INFO'
    cache_ttl:        int     = 3600
    default_capital:  float   = 10000
    default_risk:     Literal['conservative', 'moderate', 'aggressive'] = 'moderate'
    
    tickers: list[str] = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM',  'BAC',  'GS',
        'JNJ',  'PFE',  'UNH',
        'XOM',  'CVX',
        'WMT',  'PG',   'KO',
    ]

    class Config:
        env_file = '.env'

settings = Settings()