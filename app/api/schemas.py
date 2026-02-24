from pydantic import BaseModel, Field, field_validator
from typing import Literal

# ── Request schemas ────────────────────────────────────────────────────────────

class GapsRequest(BaseModel):
    portfolio: list[str] = Field(..., min_length=1, max_length=20)
    top_n:     int       = Field(default=5, ge=1, le=20)

    @field_validator('portfolio')
    @classmethod
    def uppercase_tickers(cls, v):
        return [t.strip().upper() for t in v]

class OptimizeRequest(BaseModel):
    tickers: list[str]                                    = Field(..., min_length=2, max_length=20)
    risk:    Literal['conservative', 'moderate', 'aggressive'] = 'moderate'

    @field_validator('tickers')
    @classmethod
    def uppercase_tickers(cls, v):
        return [t.strip().upper() for t in v]

class SimilarSummaryRequest(BaseModel):
    ticker:  str
    results: list[dict]

class GapsSummaryRequest(BaseModel):
    portfolio: list[str]
    results:   list[dict]

class OptimizeSummaryRequest(BaseModel):
    tickers: list[str]
    risk:    str
    result:  dict

# ── Response schemas ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str
    ready:           bool
    ticker_count:    int
    uptime_seconds:  float

class SimilarResponse(BaseModel):
    ticker:      str
    sector:      str
    similarity:  float
    beta:        float
    momentum_6m: float
    volatility:  float

class GapResponse(BaseModel):
    ticker:      str
    sector:      str
    correlation: float

class OptimizeResponse(BaseModel):
    weights:         dict[str, float]
    expected_return: float
    volatility:      float
    sharpe_ratio:    float

class SummaryResponse(BaseModel):
    summary: str

class ErrorResponse(BaseModel):
    error:   str
    detail:  str | None = None
    path:    str | None = None