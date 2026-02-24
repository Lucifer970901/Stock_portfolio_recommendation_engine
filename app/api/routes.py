import time
from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    GapsRequest, OptimizeRequest,
    SimilarSummaryRequest, GapsSummaryRequest, OptimizeSummaryRequest,
    HealthResponse, SimilarResponse, GapResponse,
    OptimizeResponse, SummaryResponse
)
from app.core.validators import validate_tickers, validate_min_tickers
from app.services.recommender import recommender
from app.services.summarizer import summarize_similar, summarize_gaps, summarize_optimize
from app.core.logger import get_logger
from app.evaluation.backtester import backtest_optimizer, compute_portfolio_metrics

log    = get_logger(__name__)
router = APIRouter(prefix='/api/v1')

START_TIME = time.time()

# ── Health ─────────────────────────────────────────────────────────────────────

@router.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status         = 'ok',
        ready          = recommender.is_ready,
        ticker_count   = len(recommender.combined_df) if recommender.is_ready else 0,
        uptime_seconds = round(time.time() - START_TIME, 1),
    )

# ── Similarity ─────────────────────────────────────────────────────────────────

@router.get('/similar/{ticker}', response_model=list[SimilarResponse])
def similar(ticker: str, top_n: int = 5) -> list[SimilarResponse]:
    ticker = ticker.strip().upper()
    universe = recommender.combined_df.index.tolist()
    validate_tickers([ticker], universe)

    try:
        results = recommender.similar(ticker, top_n)
        return [SimilarResponse(**r) for r in results]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ── Gaps ───────────────────────────────────────────────────────────────────────

@router.post('/gaps', response_model=list[GapResponse])
def gaps(req: GapsRequest) -> list[GapResponse]:
    universe = recommender.combined_df.index.tolist()
    validate_tickers(req.portfolio, universe)

    results = recommender.gaps(req.portfolio, req.top_n)
    return [GapResponse(**r) for r in results]

# ── Optimize ───────────────────────────────────────────────────────────────────

@router.post('/optimize', response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    universe = recommender.combined_df.index.tolist()
    validate_tickers(req.tickers, universe)
    validate_min_tickers(req.tickers, minimum=2)

    result = recommender.optimize(req.tickers, req.risk)
    return OptimizeResponse(**result)

# ── Optimize ───────────────────────────────────────────────────────────────────

@router.get('/evaluate/optimizer')
def evaluate_optimizer(tickers: str, risk: str = 'moderate'):
    """
    Walk-forward backtest of the portfolio optimizer.
    tickers: comma-separated e.g. AAPL,MSFT,JNJ,XOM
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    universe    = recommender.combined_df.index.tolist()
    validate_tickers(ticker_list, universe)
    validate_min_tickers(ticker_list, minimum=3)
    return backtest_optimizer(recommender.prices, ticker_list, risk)

@router.post('/evaluate/portfolio_metrics')
def portfolio_metrics(req: OptimizeRequest):
    """Compute realized metrics for a given set of weights"""
    result  = recommender.optimize(req.tickers, req.risk)
    metrics = compute_portfolio_metrics(recommender.prices, result['weights'])
    return {
        'predicted': result,
        'realized':  metrics,
        'gap': {
            'return_gap': round(result['expected_return'] - metrics['realized_annual_return'], 4),
            'vol_gap':    round(result['volatility']      - metrics['realized_volatility'],    4),
            'note':       'Positive gap = optimizer was optimistic. Expected for in-sample prediction.'
        }
    }
# ── Summarize ──────────────────────────────────────────────────────────────────

@router.post('/summarize/similar', response_model=SummaryResponse)
def summarize_similar_endpoint(req: SimilarSummaryRequest) -> SummaryResponse:
    return SummaryResponse(summary=summarize_similar(req.ticker, req.results))

@router.post('/summarize/gaps', response_model=SummaryResponse)
def summarize_gaps_endpoint(req: GapsSummaryRequest) -> SummaryResponse:
    return SummaryResponse(summary=summarize_gaps(req.portfolio, req.results))

@router.post('/summarize/optimize', response_model=SummaryResponse)
def summarize_optimize_endpoint(req: OptimizeSummaryRequest) -> SummaryResponse:
    return SummaryResponse(summary=summarize_optimize(req.tickers, req.risk, req.result))