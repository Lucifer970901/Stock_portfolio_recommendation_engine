from groq import Groq
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are a concise financial analyst assistant. 
You summarize stock recommendation results in plain English for retail investors.
Keep summaries to 3-4 sentences. Be specific, use the numbers provided.
Never give financial advice — frame everything as analytical observations.
Do not use bullet points. Write in flowing prose."""

def summarize_similar(ticker: str, results: list[dict]) -> str:
    """Summarize stock similarity results"""
    log.info(f"Summarizing similarity results for {ticker}")

    results_text = "\n".join([
        f"- {r['ticker']} ({r['sector']}): similarity {r['similarity']:.1%}, "
        f"beta {r.get('beta', 0):.2f}, 6m momentum {r.get('momentum_6m', 0):.1%}, "
        f"volatility {r.get('volatility', 0):.1%}"
        for r in results
    ])

    prompt = f"""The user searched for stocks similar to {ticker}.
Here are the top similar stocks found:

{results_text}

Summarize what these results tell us about {ticker} and its peers.
Mention sector concentration, risk profile (beta/volatility), and momentum trends."""

    return _call_groq(prompt)


def summarize_gaps(portfolio: list[str], results: list[dict]) -> str:
    """Summarize portfolio gap analysis results"""
    log.info(f"Summarizing gap analysis for {portfolio}")

    results_text = "\n".join([
        f"- {r['ticker']} ({r.get('sector', 'Unknown')}): correlation {r['correlation']:.3f}"
        for r in results
    ])

    prompt = f"""The user has a portfolio of: {', '.join(portfolio)}.
The gap analysis identified these low-correlation stocks that could improve diversification:

{results_text}

Summarize what sectors or risk factors are missing from the current portfolio,
and why the recommended stocks would improve diversification.
Mention the most compelling picks specifically."""

    return _call_groq(prompt)


def summarize_optimize(tickers: list[str], risk: str, result: dict) -> str:
    """Summarize portfolio optimization results"""
    log.info(f"Summarizing optimization for {tickers}")

    weights_text = "\n".join([
        f"- {ticker}: {weight:.1%}"
        for ticker, weight in result['weights'].items()
    ])

    prompt = f"""The user optimized a {risk} risk portfolio of: {', '.join(tickers)}.
Here are the optimized weights:

{weights_text}

Portfolio metrics:
- Expected Annual Return: {result['expected_return']:.1%}
- Annual Volatility: {result['volatility']:.1%}
- Sharpe Ratio: {result['sharpe_ratio']:.2f}

Summarize what the optimizer decided, why certain stocks got higher allocations,
and what the Sharpe ratio and return/volatility tradeoff means for a {risk} investor."""

    return _call_groq(prompt)


def _call_groq(prompt: str) -> str:
    """Make the actual Groq API call"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return f"Summary unavailable: {str(e)}"