"""
Summarizer
----------
Generates plain-English summaries of recommender results using LLM.

LLM provider strategy:
  Primary  : HuggingFace Inference API (Mistral-7B-Instruct)
             - Free tier, strong instruction following
             - Good with structured financial data
  Fallback : Groq (Llama-3.3-70b-versatile)
             - Used if HF key missing or HF API fails

Three summary types:
  - summarize_similar()  : explains similarity results and peer group
  - summarize_gaps()     : explains portfolio diversification gaps
  - summarize_optimize() : explains portfolio optimization output

All summaries are framed as analytical observations, not financial advice.
"""

from groq import Groq
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

# ── LLM clients ───────────────────────────────────────────────────────────────

HF_MODEL   = settings.hf_model
GROQ_MODEL = settings.groq_model

_groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None
_hf_client   = None

if settings.hf_api_key:
    try:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(
            model=HF_MODEL,
            token=settings.hf_api_key,
        )
        log.info(f"HuggingFace client initialised — model: {HF_MODEL}")
    except Exception as e:
        log.warning(f"HuggingFace client failed to initialise: {e}")
else:
    log.info("HF_API_KEY not set — using Groq as primary LLM")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a concise financial analyst assistant.
You summarize stock recommendation results in plain English for retail investors.
Keep summaries to 3-4 sentences. Be specific — use the numbers provided.
Never give financial advice — frame everything as analytical observations.
Do not use bullet points. Write in flowing prose.
Do not start with 'Certainly' or 'Sure' or any filler phrase."""


# ── LLM call layer ────────────────────────────────────────────────────────────

def _call_hf(prompt: str) -> str:
    """Call HuggingFace Inference API."""
    response = _hf_client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _call_groq_api(prompt: str) -> str:
    """Call Groq API."""
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def _call_groq(prompt: str, retries: int = 2) -> str:
    """
    Call the best available LLM with retry logic.

    Priority:
      1. HuggingFace Mistral-7B (if hf_api_key set)
      2. Groq Llama-3.3-70b (fallback)

    If primary fails, falls back to secondary automatically.
    Retries apply to each provider independently.

    Args:
        prompt:  user prompt to send
        retries: number of retry attempts per provider

    Returns:
        Summary string or error message
    """
    providers = []

    if _hf_client:
        providers.append(('HuggingFace', _call_hf))
    if _groq_client:
        providers.append(('Groq', _call_groq_api))

    if not providers:
        return "Summary unavailable: no LLM API keys configured (set HF_API_KEY or GROQ_API_KEY)"

    last_error = None
    for provider_name, provider_fn in providers:
        for attempt in range(retries + 1):
            try:
                result = provider_fn(prompt)
                if attempt > 0 or provider_name != providers[0][0]:
                    log.info(f"LLM call succeeded via {provider_name} "
                             f"(attempt {attempt + 1})")
                return result
            except Exception as e:
                last_error = e
                if attempt < retries:
                    log.warning(
                        f"{provider_name} attempt {attempt + 1} failed: {e}, retrying..."
                    )
                else:
                    log.warning(
                        f"{provider_name} failed after {retries + 1} attempts: {e}"
                        + (f", trying next provider" if provider_name != providers[-1][0] else "")
                    )

    log.error(f"All LLM providers failed. Last error: {last_error}")
    return f"Summary unavailable: {str(last_error)}"


# ── Public summarizer functions ───────────────────────────────────────────────

def summarize_similar(ticker: str, results: list[dict]) -> str:
    """
    Summarize stock similarity results.

    Args:
        ticker:  the query ticker
        results: list of similar stock dicts from get_similar_stocks()
    """
    log.info(f"Summarizing similarity results for {ticker}")

    results_text = "\n".join([
        f"- {r['ticker']} ({r.get('sector', 'Unknown')}, {r.get('cluster_label', '')}): "
        f"similarity {r['similarity']:.1%}, "
        f"PE {r.get('pe_ratio', 'N/A')}, "
        f"revenue growth {r.get('revenue_growth', 0):.1%}, "
        f"beta {r.get('beta', 0):.2f}, "
        f"6m momentum {r.get('momentum_6m', 0):.1%}"
        for r in results
    ])

    sectors = [r.get('sector', '') for r in results]
    dominant_sector = max(set(sectors), key=sectors.count) if sectors else 'mixed'
    sector_note = (
        f"The results are concentrated in {dominant_sector}."
        if sectors.count(dominant_sector) > len(sectors) / 2
        else "The results span multiple sectors."
    )

    prompt = f"""The user searched for stocks similar to {ticker}.
{sector_note}

Here are the top similar stocks found:
{results_text}

Summarize what these results tell us about {ticker}'s peer group.
Mention valuation (PE), growth profile, and risk characteristics (beta, momentum).
Note any cross-sector similarities that are surprising or insightful."""

    return _call_groq(prompt)


def summarize_gaps(portfolio: list[str], results: list[dict]) -> str:
    """
    Summarize portfolio gap analysis results.

    Args:
        portfolio: list of tickers in the current portfolio
        results:   list of gap recommendation dicts from recommender.gaps()
    """
    log.info(f"Summarizing gap analysis for {portfolio}")

    results_text = "\n".join([
        f"- {r['ticker']} ({r.get('sector', 'Unknown')}): "
        f"correlation with portfolio {r['correlation']:.3f}"
        for r in results
    ])

    gap_sectors = list({r.get('sector', '') for r in results if r.get('sector')})

    prompt = f"""The user has a portfolio of {len(portfolio)} stocks: {', '.join(portfolio)}.
The gap analysis identified these low-correlation stocks that could improve diversification:

{results_text}

The recommended additions span these sectors: {', '.join(gap_sectors)}.

Summarize what sectors or risk factors appear underrepresented in the current portfolio.
Explain why low correlation matters for diversification.
Highlight the most compelling 1-2 picks with specific reasoning."""

    return _call_groq(prompt)


def summarize_optimize(tickers: list[str], risk: str, result: dict) -> str:
    """
    Summarize portfolio optimization results including capital allocation.

    Args:
        tickers: candidate tickers passed to optimizer
        risk:    risk profile used
        result:  optimization result dict from optimize_portfolio()
    """
    log.info(f"Summarizing optimization for {tickers}")

    if result.get('allocation'):
        allocation_text = "\n".join([
            f"- {ticker}: {details['weight']:.1%} "
            f"({details['shares']} shares @ ${details['price']:.2f} "
            f"= ${details['allocation']:,.0f})"
            for ticker, details in sorted(
                result['allocation'].items(),
                key=lambda x: -x[1]['allocation']
            )
        ])
    else:
        allocation_text = "\n".join([
            f"- {ticker}: {weight:.1%}"
            for ticker, weight in result['weights'].items()
        ])

    capital     = result.get('capital', 10000)
    leftover    = result.get('leftover_cash', 0)
    n_positions = result.get('n_positions', len(result['weights']))

    prompt = f"""The user ran a {risk} risk portfolio optimization with ${capital:,.0f} capital.
The optimizer selected {n_positions} positions from {len(tickers)} candidates,
leaving ${leftover:,.2f} in uninvested cash.

Capital allocation:
{allocation_text}

Portfolio metrics:
- Expected Annual Return: {result['expected_return']:.1%}
- Annual Volatility: {result['volatility']:.1%}
- Sharpe Ratio: {result['sharpe_ratio']:.2f}

Summarize why the optimizer concentrated on these specific stocks,
what the sector mix tells us about the {risk} risk strategy,
and what the Sharpe ratio and return/volatility tradeoff means in practice.
Note if any allocation seems surprising and why."""

    return _call_groq(prompt)