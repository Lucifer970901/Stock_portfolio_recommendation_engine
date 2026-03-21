"""
Tests for app/api/routes.py

Uses FastAPI TestClient with recommender mocked out — no real data fetching.
Tests cover HTTP status codes, response shapes, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, PropertyMock

# ── App setup ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_recommender():
    """Mock recommender service with is_ready=True and sample data."""
    mock = MagicMock()
    mock.is_ready = True
    mock.combined_df.index.tolist.return_value = ['AAPL', 'MSFT', 'JNJ', 'XOM', 'JPM']

    # similar() response
    mock.similar.return_value = [
        {
            'ticker':      'MSFT',
            'sector':      'Technology',
            'similarity':  0.85,
            'beta':        1.1,
            'momentum_6m': -0.10,
            'volatility':  0.24,
        }
    ]

    # gaps() response
    mock.gaps.return_value = [
        {'ticker': 'JNJ', 'sector': 'Healthcare', 'correlation': -0.05}
    ]

    # optimize() response
    mock.optimize.return_value = {
        'weights':         {'AAPL': 0.5, 'MSFT': 0.5},
        'allocation':      {
            'AAPL': {'shares': 5, 'price': 200.0, 'allocation': 1000.0, 'weight': 0.5},
            'MSFT': {'shares': 3, 'price': 380.0, 'allocation': 1140.0, 'weight': 0.5},
        },
        'leftover_cash':   0.0,
        'expected_return': 0.15,
        'volatility':      0.12,
        'sharpe_ratio':    0.83,
        'capital':         10000,
        'risk_profile':    'moderate',
        'n_positions':     2,
    }

    return mock


@pytest.fixture
def client(mock_recommender):
    """TestClient with mocked recommender."""
    import app.api.routes as routes_module
    with patch.object(routes_module, 'recommender', mock_recommender):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    """GET /health should return 200."""
    response = client.get('/api/v1/health')
    assert response.status_code == 200


def test_health_response_shape(client):
    """Health response should have status, ready, ticker_count, uptime_seconds."""
    response = client.get('/api/v1/health')
    data     = response.json()
    assert 'status'          in data
    assert 'ready'           in data
    assert 'ticker_count'    in data
    assert 'uptime_seconds'  in data


def test_health_status_ok(client):
    """Health status should be 'ok'."""
    response = client.get('/api/v1/health')
    assert response.json()['status'] == 'ok'


def test_health_ready_true(client):
    """Health ready should be True when recommender is built."""
    response = client.get('/api/v1/health')
    assert response.json()['ready'] is True


# ── /similar/{ticker} ─────────────────────────────────────────────────────────

def test_similar_returns_200(client):
    """GET /similar/AAPL should return 200."""
    response = client.get('/api/v1/similar/AAPL')
    assert response.status_code == 200


def test_similar_returns_list(client):
    """Similar endpoint should return a list."""
    response = client.get('/api/v1/similar/AAPL')
    assert isinstance(response.json(), list)


def test_similar_response_shape(client):
    """Each similar result should have required fields."""
    response = client.get('/api/v1/similar/AAPL')
    result   = response.json()[0]
    for field in ['ticker', 'sector', 'similarity', 'beta', 'momentum_6m', 'volatility']:
        assert field in result


def test_similar_top_n_param(client, mock_recommender):
    """top_n query param should be passed to recommender."""
    client.get('/api/v1/similar/AAPL?top_n=3')
    mock_recommender.similar.assert_called_with('AAPL', 3)


def test_similar_invalid_ticker_returns_400(client):
    """Unknown ticker should return 400."""
    response = client.get('/api/v1/similar/FAKE_TICKER')
    assert response.status_code == 400


def test_similar_ticker_uppercased(client, mock_recommender):
    """Ticker should be uppercased before calling recommender."""
    client.get('/api/v1/similar/aapl')
    mock_recommender.similar.assert_called_with('AAPL', 5)


# ── /gaps ─────────────────────────────────────────────────────────────────────

def test_gaps_returns_200(client):
    """POST /gaps should return 200."""
    response = client.post('/api/v1/gaps', json={'portfolio': ['AAPL', 'MSFT']})
    assert response.status_code == 200


def test_gaps_returns_list(client):
    """Gaps endpoint should return a list."""
    response = client.post('/api/v1/gaps', json={'portfolio': ['AAPL', 'MSFT']})
    assert isinstance(response.json(), list)


def test_gaps_response_shape(client):
    """Each gap result should have ticker, sector, correlation."""
    response = client.post('/api/v1/gaps', json={'portfolio': ['AAPL', 'MSFT']})
    result   = response.json()[0]
    for field in ['ticker', 'sector', 'correlation']:
        assert field in result


def test_gaps_invalid_ticker_returns_400(client):
    """Unknown ticker in portfolio should return 400."""
    response = client.post('/api/v1/gaps', json={'portfolio': ['FAKE_TICKER']})
    assert response.status_code == 400


def test_gaps_empty_portfolio_returns_422(client):
    """Empty portfolio should return 422 (Pydantic validation)."""
    response = client.post('/api/v1/gaps', json={'portfolio': []})
    assert response.status_code == 422


def test_gaps_top_n_param(client, mock_recommender):
    """top_n should be passed to recommender."""
    client.post('/api/v1/gaps', json={'portfolio': ['AAPL', 'MSFT'], 'top_n': 3})
    mock_recommender.gaps.assert_called_with(['AAPL', 'MSFT'], 3)


# ── /optimize ─────────────────────────────────────────────────────────────────

def test_optimize_returns_200(client):
    """POST /optimize should return 200."""
    response = client.post(
        '/api/v1/optimize',
        json={'tickers': ['AAPL', 'MSFT'], 'risk': 'moderate'}
    )
    assert response.status_code == 200


def test_optimize_response_shape(client):
    """Optimize response should have weights, return, volatility, sharpe."""
    response = client.post(
        '/api/v1/optimize',
        json={'tickers': ['AAPL', 'MSFT'], 'risk': 'moderate'}
    )
    data = response.json()
    for field in ['weights', 'expected_return', 'volatility', 'sharpe_ratio']:
        assert field in data


def test_optimize_invalid_ticker_returns_400(client):
    """Unknown ticker should return 400."""
    response = client.post(
        '/api/v1/optimize',
        json={'tickers': ['FAKE', 'MSFT'], 'risk': 'moderate'}
    )
    assert response.status_code == 400


def test_optimize_single_ticker_returns_422(client):
    """Single ticker should return 422 (Pydantic min_length=2)."""
    response = client.post(
        '/api/v1/optimize',
        json={'tickers': ['AAPL'], 'risk': 'moderate'}
    )
    assert response.status_code == 422


def test_optimize_invalid_risk_returns_422(client):
    """Invalid risk level should return 422."""
    response = client.post(
        '/api/v1/optimize',
        json={'tickers': ['AAPL', 'MSFT'], 'risk': 'yolo'}
    )
    assert response.status_code == 422


def test_optimize_default_risk_moderate(client, mock_recommender):
    """Default risk should be moderate."""
    client.post('/api/v1/optimize', json={'tickers': ['AAPL', 'MSFT']})
    call_args = mock_recommender.optimize.call_args
    assert call_args[0][1] == 'moderate' or call_args[1].get('risk') == 'moderate'


def test_optimize_tickers_uppercased(client, mock_recommender):
    """Tickers should be uppercased by Pydantic validator."""
    client.post(
        '/api/v1/optimize',
        json={'tickers': ['aapl', 'msft'], 'risk': 'moderate'}
    )
    call_args = mock_recommender.optimize.call_args[0][0]
    assert 'AAPL' in call_args
    assert 'MSFT' in call_args


# ── Schema validation tests ───────────────────────────────────────────────────

def test_gaps_request_uppercase_validator():
    """GapsRequest should uppercase tickers automatically."""
    from app.api.schemas import GapsRequest
    req = GapsRequest(portfolio=['aapl', 'msft'])
    assert req.portfolio == ['AAPL', 'MSFT']


def test_gaps_request_strips_whitespace():
    """GapsRequest should strip whitespace from tickers."""
    from app.api.schemas import GapsRequest
    req = GapsRequest(portfolio=[' AAPL ', ' MSFT'])
    assert req.portfolio == ['AAPL', 'MSFT']


def test_optimize_request_uppercase_validator():
    """OptimizeRequest should uppercase tickers automatically."""
    from app.api.schemas import OptimizeRequest
    req = OptimizeRequest(tickers=['aapl', 'msft'])
    assert req.tickers == ['AAPL', 'MSFT']


def test_optimize_request_default_risk():
    """OptimizeRequest default risk should be moderate."""
    from app.api.schemas import OptimizeRequest
    req = OptimizeRequest(tickers=['AAPL', 'MSFT'])
    assert req.risk == 'moderate'


def test_optimize_request_invalid_risk_raises():
    """OptimizeRequest should reject invalid risk values."""
    from pydantic import ValidationError
    from app.api.schemas import OptimizeRequest
    with pytest.raises(ValidationError):
        OptimizeRequest(tickers=['AAPL', 'MSFT'], risk='yolo')


def test_health_response_schema():
    """HealthResponse should accept valid data."""
    from app.api.schemas import HealthResponse
    resp = HealthResponse(
        status='ok', ready=True, ticker_count=50, uptime_seconds=1.5
    )
    assert resp.status == 'ok'


def test_similar_response_schema():
    """SimilarResponse should accept valid data."""
    from app.api.schemas import SimilarResponse
    resp = SimilarResponse(
        ticker='MSFT', sector='Technology',
        similarity=0.85, beta=1.1, momentum_6m=-0.1, volatility=0.24
    )
    assert resp.ticker == 'MSFT'


def test_optimize_response_schema():
    """OptimizeResponse should accept valid data."""
    from app.api.schemas import OptimizeResponse
    resp = OptimizeResponse(
        weights={'AAPL': 0.5, 'MSFT': 0.5},
        expected_return=0.15,
        volatility=0.12,
        sharpe_ratio=0.83,
    )
    assert resp.sharpe_ratio == 0.83