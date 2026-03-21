"""
Tests for app/services/summarizer.py

Strategy:
  - All LLM calls are mocked — no real API calls made
  - Tests verify prompt construction, retry logic, provider fallback,
    and correct handling of the new optimize result format
"""

import pytest
from unittest.mock import patch, MagicMock
from app.models.summarizer import (
    summarize_similar,
    summarize_gaps,
    summarize_optimize,
    _call_groq,
)

# ── Shared test data ──────────────────────────────────────────────────────────

SIMILAR_RESULTS = [
    {
        'ticker':         'NVDA',
        'sector':         'Technology',
        'cluster_label':  'Hypergrowth',
        'similarity':     0.675,
        'pe_ratio':       35.0,
        'revenue_growth': 1.14,
        'beta':           2.31,
        'momentum_6m':    0.084,
        'volatility':     0.496,
    },
    {
        'ticker':         'GOOGL',
        'sector':         'Communication Services',
        'cluster_label':  'Quality Growth',
        'similarity':     0.296,
        'pe_ratio':       13.26,
        'revenue_growth': 0.139,
        'beta':           1.09,
        'momentum_6m':    0.510,
        'volatility':     0.298,
    },
    {
        'ticker':         'MSFT',
        'sector':         'Technology',
        'cluster_label':  'Quality Growth',
        'similarity':     0.150,
        'pe_ratio':       23.78,
        'revenue_growth': 0.149,
        'beta':           1.08,
        'momentum_6m':    -0.230,
        'volatility':     0.240,
    },
]

GAP_RESULTS = [
    {'ticker': 'JNJ', 'sector': 'Healthcare',         'correlation': -0.052},
    {'ticker': 'KO',  'sector': 'Consumer Defensive', 'correlation':  0.010},
    {'ticker': 'XOM', 'sector': 'Energy',             'correlation':  0.161},
]

OPTIMIZE_RESULT = {
    'weights':         {'AAPL': 0.031, 'MSFT': 0.284, 'JNJ': 0.469, 'XOM': 0.216},
    'allocation': {
        'JNJ':  {'shares': 9,  'price': 235.37, 'allocation': 2118.33, 'weight': 0.469},
        'MSFT': {'shares': 7,  'price': 381.87, 'allocation': 2673.09, 'weight': 0.284},
        'XOM':  {'shares': 13, 'price': 159.67, 'allocation': 2075.71, 'weight': 0.216},
        'AAPL': {'shares': 1,  'price': 227.48, 'allocation':  227.48, 'weight': 0.031},
    },
    'leftover_cash':   5.39,
    'expected_return': 0.1823,
    'volatility':      0.1209,
    'sharpe_ratio':    1.0939,
    'capital':         10000,
    'risk_profile':    'moderate',
    'n_positions':     4,
}

MOCK_SUMMARY = "This is a mocked financial summary for testing purposes."


# ── Helper ────────────────────────────────────────────────────────────────────

def make_mock_groq_response(text: str):
    mock_choice                 = MagicMock()
    mock_choice.message.content = text
    mock_response               = MagicMock()
    mock_response.choices       = [mock_choice]
    return mock_response


def make_mock_hf_response(text: str):
    mock_choice                 = MagicMock()
    mock_choice.message.content = text
    mock_response               = MagicMock()
    mock_response.choices       = [mock_choice]
    return mock_response


# ── _call_groq unit tests ─────────────────────────────────────────────────────

def test_call_groq_returns_string():
    """_call_groq should return a non-empty string."""
    with patch('app.models.summarizer._hf_client') as mock_hf:
        mock_hf.chat_completion.return_value = make_mock_hf_response(MOCK_SUMMARY)
        result = _call_groq("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0


def test_call_groq_returns_stripped_text():
    """_call_groq should strip whitespace from response."""
    with patch('app.models.summarizer._hf_client') as mock_hf:
        mock_hf.chat_completion.return_value = make_mock_hf_response("  padded  ")
        result = _call_groq("test prompt")
        assert result == "padded"


def test_call_groq_handles_api_error():
    """_call_groq should return error string when all providers fail."""
    with patch('app.models.summarizer._hf_client') as mock_hf, \
         patch('app.models.summarizer._groq_client') as mock_groq:
        mock_hf.chat_completion.side_effect   = Exception("HF error")
        mock_groq.chat.completions.create.side_effect = Exception("Groq error")
        result = _call_groq("test prompt", retries=0)
        assert isinstance(result, str)
        assert "unavailable" in result.lower() or "error" in result.lower()


def test_call_groq_retries_on_failure():
    """_call_groq should retry on failure then succeed."""
    with patch('app.models.summarizer._hf_client') as mock_hf:
        mock_hf.chat_completion.side_effect = [
            Exception("transient error"),
            Exception("transient error"),
            make_mock_hf_response(MOCK_SUMMARY),
        ]
        result = _call_groq("test prompt", retries=2)
        assert result == MOCK_SUMMARY
        assert mock_hf.chat_completion.call_count == 3


def test_call_groq_falls_back_to_groq():
    """_call_groq should fall back to Groq when HF fails."""
    with patch('app.models.summarizer._hf_client') as mock_hf, \
         patch('app.models.summarizer._groq_client') as mock_groq:
        mock_hf.chat_completion.side_effect = Exception("HF unavailable")
        mock_groq.chat.completions.create.return_value = make_mock_groq_response(MOCK_SUMMARY)
        result = _call_groq("test prompt", retries=0)
        assert result == MOCK_SUMMARY
        mock_groq.chat.completions.create.assert_called()


def test_call_groq_no_providers():
    """_call_groq should return helpful message when no providers configured."""
    with patch('app.models.summarizer._hf_client', None), \
         patch('app.models.summarizer._groq_client', None):
        result = _call_groq("test prompt")
        assert "unavailable" in result.lower()
        assert "api key" in result.lower() or "configured" in result.lower()


# ── summarize_similar tests ───────────────────────────────────────────────────

def test_summarize_similar_returns_string():
    """summarize_similar should return a non-empty string."""
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_similar('AAPL', SIMILAR_RESULTS)
        assert isinstance(result, str) and len(result) > 0


def test_summarize_similar_includes_ticker():
    """Query ticker should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'AAPL' in prompt


def test_summarize_similar_includes_all_result_tickers():
    """All result tickers should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        for r in SIMILAR_RESULTS:
            assert r['ticker'] in prompt


def test_summarize_similar_includes_cluster_label():
    """cluster_label should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'Hypergrowth' in prompt or 'Quality Growth' in prompt


def test_summarize_similar_includes_revenue_growth():
    """Revenue growth values should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'revenue' in prompt.lower() or '1.14' in prompt


def test_summarize_similar_handles_empty_results():
    """summarize_similar should handle empty result list without crashing."""
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_similar('AAPL', [])
        assert isinstance(result, str)


def test_summarize_similar_called_once():
    """_call_groq should be called exactly once."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        assert mock_call.call_count == 1


def test_summarize_similar_sector_concentration_noted():
    """Sector concentration should be detected and noted in prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_similar('AAPL', SIMILAR_RESULTS)
        prompt = mock_call.call_args[0][0]
        # Two of three results are Technology — should be flagged
        assert 'Technology' in prompt


# ── summarize_gaps tests ──────────────────────────────────────────────────────

def test_summarize_gaps_returns_string():
    """summarize_gaps should return a non-empty string."""
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        assert isinstance(result, str) and len(result) > 0


def test_summarize_gaps_includes_portfolio_tickers():
    """Portfolio tickers should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'AAPL' in prompt and 'MSFT' in prompt


def test_summarize_gaps_includes_correlations():
    """Correlation values should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'correlation' in prompt.lower() or '-0.052' in prompt


def test_summarize_gaps_includes_recommended_tickers():
    """Recommended tickers should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        for r in GAP_RESULTS:
            assert r['ticker'] in prompt


def test_summarize_gaps_includes_gap_sectors():
    """Sectors from gap results should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_gaps(['AAPL', 'MSFT'], GAP_RESULTS)
        prompt = mock_call.call_args[0][0]
        assert 'Healthcare' in prompt or 'Energy' in prompt


def test_summarize_gaps_single_ticker_portfolio():
    """summarize_gaps should handle single-ticker portfolio."""
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_gaps(['AAPL'], GAP_RESULTS)
        assert isinstance(result, str)


# ── summarize_optimize tests ──────────────────────────────────────────────────

def test_summarize_optimize_returns_string():
    """summarize_optimize should return a non-empty string."""
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        assert isinstance(result, str) and len(result) > 0


def test_summarize_optimize_includes_risk():
    """Risk level should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT'], 'aggressive', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert 'aggressive' in prompt.lower()


def test_summarize_optimize_includes_sharpe_and_return():
    """Sharpe ratio and return should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert '1.09' in prompt or 'sharpe' in prompt.lower()
        assert '18.2' in prompt or 'return' in prompt.lower()


def test_summarize_optimize_includes_all_weight_tickers():
    """All tickers in weights should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        for ticker in OPTIMIZE_RESULT['weights']:
            assert ticker in prompt


def test_summarize_optimize_includes_capital():
    """Capital amount should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert '10,000' in prompt or '10000' in prompt


def test_summarize_optimize_includes_dollar_allocation():
    """Dollar allocation amounts should appear in the prompt."""
    with patch('app.models.summarizer._call_groq') as mock_call:
        mock_call.return_value = MOCK_SUMMARY
        summarize_optimize(['AAPL', 'MSFT', 'JNJ', 'XOM'], 'moderate', OPTIMIZE_RESULT)
        prompt = mock_call.call_args[0][0]
        assert '$' in prompt or 'shares' in prompt.lower()


def test_summarize_optimize_all_risk_levels():
    """summarize_optimize should work for all three risk levels."""
    for risk in ['conservative', 'moderate', 'aggressive']:
        with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
            result = summarize_optimize(['AAPL', 'MSFT'], risk, OPTIMIZE_RESULT)
            assert isinstance(result, str)


def test_summarize_optimize_handles_missing_allocation():
    """summarize_optimize should handle result without allocation key."""
    result_no_alloc = {
        'weights':         {'AAPL': 0.5, 'MSFT': 0.5},
        'expected_return': 0.15,
        'volatility':      0.12,
        'sharpe_ratio':    0.80,
        'capital':         10000,
        'n_positions':     2,
    }
    with patch('app.models.summarizer._call_groq', return_value=MOCK_SUMMARY):
        result = summarize_optimize(['AAPL', 'MSFT'], 'moderate', result_no_alloc)
        assert isinstance(result, str)


def test_summarize_optimize_handles_api_error():
    """summarize_optimize should return error string on API failure."""
    with patch('app.models.summarizer._call_groq',
               return_value="Summary unavailable: timeout"):
        result = summarize_optimize(['AAPL', 'MSFT'], 'moderate', OPTIMIZE_RESULT)
        assert isinstance(result, str) and len(result) > 0


# ── Config-driven model tests ─────────────────────────────────────────────────

def test_hf_model_comes_from_config():
    """HF_MODEL should be read from settings, not hardcoded."""
    import app.models.summarizer as summarizer_module
    from app.core.config import settings
    assert summarizer_module.HF_MODEL == settings.hf_model


def test_groq_model_comes_from_config():
    """GROQ_MODEL should be read from settings, not hardcoded."""
    import app.models.summarizer as summarizer_module
    from app.core.config import settings
    assert summarizer_module.GROQ_MODEL == settings.groq_model