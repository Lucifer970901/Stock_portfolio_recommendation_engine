"""
Tests for app/core/validators.py
"""

import pytest
from fastapi import HTTPException
from app.core.validators import validate_tickers, validate_min_tickers, validate_risk

UNIVERSE = ['AAPL', 'MSFT', 'JNJ', 'XOM', 'JPM']


# ── validate_tickers ──────────────────────────────────────────────────────────

def test_validate_tickers_valid():
    """Valid tickers should pass without raising."""
    result = validate_tickers(['AAPL', 'MSFT'], UNIVERSE)
    assert result == ['AAPL', 'MSFT']


def test_validate_tickers_single_valid():
    """Single valid ticker should pass."""
    result = validate_tickers(['AAPL'], UNIVERSE)
    assert result == ['AAPL']


def test_validate_tickers_all_universe():
    """Full universe should pass."""
    result = validate_tickers(UNIVERSE, UNIVERSE)
    assert result == UNIVERSE


def test_validate_tickers_invalid_raises():
    """Unknown ticker should raise HTTPException 400."""
    with pytest.raises(HTTPException) as exc:
        validate_tickers(['FAKE'], UNIVERSE)
    assert exc.value.status_code == 400


def test_validate_tickers_invalid_detail_contains_ticker():
    """Error detail should name the invalid ticker."""
    with pytest.raises(HTTPException) as exc:
        validate_tickers(['FAKE'], UNIVERSE)
    assert 'FAKE' in str(exc.value.detail)


def test_validate_tickers_mixed_raises():
    """Mix of valid and invalid should raise for the invalid one."""
    with pytest.raises(HTTPException) as exc:
        validate_tickers(['AAPL', 'FAKE'], UNIVERSE)
    assert exc.value.status_code == 400
    assert 'FAKE' in str(exc.value.detail)


def test_validate_tickers_empty_raises():
    """Empty ticker list with unknown entry should raise."""
    with pytest.raises(HTTPException):
        validate_tickers([''], UNIVERSE)


def test_validate_tickers_multiple_invalid():
    """Multiple invalid tickers should all appear in error detail."""
    with pytest.raises(HTTPException) as exc:
        validate_tickers(['FAKE1', 'FAKE2'], UNIVERSE)
    detail = str(exc.value.detail)
    assert 'FAKE1' in detail
    assert 'FAKE2' in detail


# ── validate_min_tickers ──────────────────────────────────────────────────────

def test_validate_min_tickers_exact_minimum():
    """Exactly the minimum number should pass."""
    validate_min_tickers(['AAPL', 'MSFT'], minimum=2)


def test_validate_min_tickers_above_minimum():
    """More than minimum should pass."""
    validate_min_tickers(['AAPL', 'MSFT', 'JNJ'], minimum=2)


def test_validate_min_tickers_below_raises():
    """Fewer than minimum should raise HTTPException 400."""
    with pytest.raises(HTTPException) as exc:
        validate_min_tickers(['AAPL'], minimum=2)
    assert exc.value.status_code == 400


def test_validate_min_tickers_empty_raises():
    """Empty list should raise."""
    with pytest.raises(HTTPException):
        validate_min_tickers([], minimum=1)


def test_validate_min_tickers_detail_contains_count():
    """Error detail should mention required count."""
    with pytest.raises(HTTPException) as exc:
        validate_min_tickers(['AAPL'], minimum=3)
    assert '3' in str(exc.value.detail)


def test_validate_min_tickers_default_minimum():
    """Default minimum should be 2."""
    with pytest.raises(HTTPException):
        validate_min_tickers(['AAPL'])


# ── validate_risk ─────────────────────────────────────────────────────────────

def test_validate_risk_conservative():
    """conservative should be valid."""
    validate_risk('conservative')


def test_validate_risk_moderate():
    """moderate should be valid."""
    validate_risk('moderate')


def test_validate_risk_aggressive():
    """aggressive should be valid."""
    validate_risk('aggressive')


def test_validate_risk_invalid_raises():
    """Unknown risk level should raise HTTPException 400."""
    with pytest.raises(HTTPException) as exc:
        validate_risk('extreme')
    assert exc.value.status_code == 400


def test_validate_risk_invalid_detail_contains_value():
    """Error detail should name the invalid risk value."""
    with pytest.raises(HTTPException) as exc:
        validate_risk('yolo')
    assert 'yolo' in str(exc.value.detail)


def test_validate_risk_case_sensitive():
    """Risk validation should be case-sensitive."""
    with pytest.raises(HTTPException):
        validate_risk('Moderate')


def test_validate_risk_empty_raises():
    """Empty string should raise."""
    with pytest.raises(HTTPException):
        validate_risk('')