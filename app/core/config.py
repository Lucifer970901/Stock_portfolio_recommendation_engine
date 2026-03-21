from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    app_env:         str   = "development"
    app_port:        int   = 8000
    log_level:       str   = "INFO"
    cache_ttl:       int   = 3600
    default_capital: float = 10_000
    default_risk:    Literal["conservative", "moderate", "aggressive"] = "moderate"
    groq_api_key:    str   = ""
    hf_api_key:      str   = ""
    hf_model:        str   = "human-centered-summarization/financial-summarization-pegasus"
    groq_model:      str   = "llama-3.3-70b-versatile"

    tickers: list[str] = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
        "CRM",  "AMD",  "INTC", "QCOM", "TXN",

        # Financials
        "JPM",  "BAC",  "GS",   "MS",   "WFC",
        "BLK",  "AXP",  "SCHW", "C",    "USB",

        # Healthcare
        "JNJ",  "PFE",  "UNH",  "ABBV", "MRK",
        "TMO",  "ABT",  "DHR",  "BMY",  "LLY",

        # Energy
        "XOM",  "CVX",  "COP",  "SLB",  "EOG",

        # Consumer
        "WMT",  "PG",   "KO",   "PEP",  "COST",
        "MCD",  "NKE",  "SBUX", "TGT",  "HD",
    ]


settings = Settings()