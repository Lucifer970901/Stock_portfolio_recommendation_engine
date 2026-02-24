from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = ConfigDict(env_file='.env') 
    app_env:          str     = 'development'
    app_port:         int     = 8000
    log_level:        str     = 'INFO'
    cache_ttl:        int     = 3600
    default_capital:  float   = 10000
    default_risk:     Literal['conservative', 'moderate', 'aggressive'] = 'moderate'
    groq_api_key:    str   = ''
    
    tickers: list[str] = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM',  'BAC',  'GS',
        'JNJ',  'PFE',  'UNH',
        'XOM',  'CVX',
        'WMT',  'PG',   'KO',
    ]


settings = Settings()