from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.recommender import recommender

router = APIRouter(prefix='/api/v1')

class OptimizeRequest(BaseModel):
    tickers: list[str]
    risk:    str = 'moderate'

class GapsRequest(BaseModel):
    portfolio: list[str]
    top_n:     int = 5

@router.get('/health')
def health():
    return {'status': 'ok', 'ready': recommender.is_ready}

@router.get('/similar/{ticker}')
def similar(ticker: str, top_n: int = 5):
    try:
        return recommender.similar(ticker.upper(), top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post('/gaps')
def gaps(req: GapsRequest):
    return recommender.gaps(
        [t.upper() for t in req.portfolio], req.top_n
    )

@router.post('/optimize')
def optimize(req: OptimizeRequest):
    return recommender.optimize(
        [t.upper() for t in req.tickers], req.risk
    )