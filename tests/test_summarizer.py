import pytest
from unittest.mock import patch, MagicMock
from app.services.summarizer import (
    summarize_similar,
    summarize_gaps,
    summarize_optimize,
    _call_groq,
)


# ── Shared test data ───────────────────────────────────────────────────────────

SIMILAR_RESULTS = [
    {'ticker': 'NVDA', 'sector': 'Technology',             'similarity': 0.675, 'beta': 2.31, 'momentum_6m': 0.084,  'volatility': 0.496},
    {'ticker': 'GOOGL','sector': 'Communication Services', 'similarity': 0.296, 'beta': 1.09, 'momentum_6m': 0.510,  'volatility': 0.298},
    {'ticker': 'MSFT', 'sector': 'Technology',             'similarity': 0.150, 'beta': 1.08, 'momentum_6m': -0.230, 'volatility': 0.240},
]

GAP_RESULTS = [
    {'ticker': 'JNJ', 'sector': 'Healthcare', 'correlation': -0.052},
    {'ticker': 'KO',  'sector': 'Consumer',   'correlation':  0.010},
    {'ticker': 'XOM', 'sector': 'Energy',     'correlation':  0.161},
]

OPTIMIZE_RESULT = {
    'weights':         {'AAPL': 0.031, 'MSFT': 0.284, 'JNJ': 0.469, 'XOM': 0.216},
    'expected_return': 0.1823,
    'volatility':      0.1209,
    'sharpe_ratio':    1.0939,
}

MOCK_SUMMARY = "This is a mocked financial summary for testing purposes."


# ── Helper: mock Groq response ─────────────────────────────────────────────────

def make_mock_groq_response(text: str):
    mock_choice          = MagicMock()
    mock_choice.message.content = text
    mock_response        = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ── _call_groq tests ───────────────────────────────────────────────────────────

def test_call_groq_returns_string():
    """_call_groq should return a string"""
    with patch('app.services.summarizer.client') as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_groq_response(MOCK_SUMMARY)
        result = _call_groq("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0


def test_call_groq_returns_stripped_text():
    """_call_groq should strip whitespace from response"""
    with patch('app.services.summarizer.client') as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_groq_response(
            "  padded summary  "
        )
        result = _call_groq("test prompt")
        assert result == "padded summary"


def test_call_groq_handles_api_error():
    """_call_groq should return error string on API failure, not raise"""
    with patch('app.services.summarizer.client') as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        result = _call_groq("test prompt")
        assert "unavailable" in result.lower() or "error" in result.lower()


def test_call_groq_uses_correct_model():
    """_call_groq should call Groq with llama3 model"""
    with patch('app.services.summarizer.client') as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_groq_response(MOCK_SUMMARY)
        _call_groq("test prompt")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert 'llama' in call_kwargs['model'].lower()


def test_call_groq_sends_system_prompt():
    """_call_groq should include a system message"""
    with patch('app.services.summarizer.client') as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_groq_response(MOCK_SUMMARY)
        _call_groq("user prompt")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages    = call_kwargs['messages']
        roles       = [m['role'] for m in messages]
        assert 'system' in roles
        assert 'user'   in roles


# ── summarize_similar tests ────────────────────────────────────────────────────

def test_summarize_similar_returns_string():
    """summarize_similar should return a non-empty string"""
    with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_similar('AAPL', SIMILAR_RESULTS)
        assert isinstance(result, str)
        assert len(result) > 0


def test_summarize_similar_includes_ticker_in_prompt():
    """summarize_similar should pass the seed ticker to the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'AAPL' in prompt


def test_summarize_similar_includes_all_tickers_in_prompt():
    """all result tickers should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        for r in SIMILAR_RESULTS:
            assert r['ticker'] in prompt


def test_summarize_similar_includes_metrics_in_prompt():
    """similarity scores and momentum should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'similarity' in prompt.lower() or '0.675' in prompt
        assert 'momentum'   in prompt.lower() or '0.084' in prompt


def test_summarize_similar_handles_empty_results():
    """summarize_similar should handle empty result list without crashing"""
    with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_similar('AAPL', [])
        assert isinstance(result, str)


def test_summarize_similar_called_once():
    """_call_groq should be called exactly once per summarize_similar call"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        assert mock_call.call_count == 1


# ── summarize_gaps tests ───────────────────────────────────────────────────────

def test_summarize_gaps_returns_string():
    """summarize_gaps should return a non-empty string"""
    with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        assert isinstance(result, str)
        assert len(result) > 0


def test_summarize_gaps_includes_portfolio_in_prompt():
    """portfolio tickers should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'AAPL' in prompt
        assert 'MSFT' in prompt


def test_summarize_gaps_includes_correlations_in_prompt():
    """correlation values should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'correlation' in prompt.lower() or '-0.052' in prompt


def test_summarize_gaps_includes_recommended_tickers():
    """recommended gap tickers should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        for r in GAP_RESULTS:
            assert r['ticker'] in prompt


def test_summarize_gaps_single_ticker_portfolio():
    """summarize_gaps should handle single-ticker portfolio"""
    with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_gaps(['AAPL'], GAP_RESULTS)
        assert isinstance(result, str)


# ── summarize_optimize tests ───────────────────────────────────────────────────

def test_summarize_optimize_returns_string():
    """summarize_optimize should return a non-empty string"""
    with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        assert isinstance(result, str)
        assert len(result) > 0


def test_summarize_optimize_includes_risk_in_prompt():
    """risk level should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT'], 'aggressive', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert 'aggressive' in prompt.lower()


def test_summarize_optimize_includes_metrics_in_prompt():
    """sharpe ratio and return should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert '1.09' in prompt or 'sharpe' in prompt.lower()
        assert '18.2' in prompt or 'return' in prompt.lower()


def test_summarize_optimize_includes_weights_in_prompt():
    """portfolio weights should appear in the prompt"""
    with patch('app.services.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        for ticker in OPTIMIZE_RESULT['weights']:
            assert ticker in prompt


def test_summarize_optimize_all_risk_levels():
    """summarize_optimize should work for all three risk levels"""
    for risk in ['conservative', 'moderate', 'aggressive']:
        with patch('app.services.summarizer._call_groq', return_value=MOCK_SUMMARY):
            result = summarize_optimize(['AAPL', 'MSFT'], risk, OPTIMIZE_RESULT)
            assert isinstance(result, str)


def test_summarize_optimize_handles_api_error():
    """summarize_optimize should return error string on API failure"""
    with patch('app.services.summarizer._call_groq', return_value="Summary unavailable: timeout"):
        result = summarize_optimize(['AAPL', 'MSFT'], 'moderate', OPTIMIZE_RESULT)
        assert isinstance(result, str)
        assert len(result) > 0