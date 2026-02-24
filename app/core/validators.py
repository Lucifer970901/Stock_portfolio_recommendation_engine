from fastapi import HTTPException
from app.core.logger import get_logger

log = get_logger(__name__)

def validate_tickers(tickers: list[str], universe: list[str]) -> list[str]:
    """Validate all tickers exist in the known universe"""
    invalid = [t for t in tickers if t not in universe]
    if invalid:
        log.warning(f"Invalid tickers requested: {invalid}")
        raise HTTPException(
            status_code=400,
            detail={
                "error":   "Unknown tickers",
                "invalid": invalid,
                "valid":   universe,
            }
        )
    return tickers

def validate_min_tickers(tickers: list[str], minimum: int = 2):
    """Ensure enough tickers for meaningful analysis"""
    if len(tickers) < minimum:
        raise HTTPException(
            status_code=400,
            detail=f"At least {minimum} tickers required, got {len(tickers)}"
        )

def validate_risk(risk: str):
    valid = {'conservative', 'moderate', 'aggressive'}
    if risk not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk level '{risk}'. Must be one of: {valid}"
        )