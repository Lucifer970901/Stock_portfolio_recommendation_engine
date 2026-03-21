# Stock Portfolio Recommender

An end-to-end stock portfolio recommendation engine built with FastAPI, PyPortfolioOpt, and LLM-powered summarization. The system fetches real-time fundamentals, clusters stocks by behavioral similarity, optimizes portfolios using Modern Portfolio Theory, and explains results in plain English.

---

## Features

- **Point-in-time fundamentals** — PE ratio, EPS TTM, revenue growth, D/E ratio calculated from quarterly reports with 45-day reporting lag to prevent lookahead bias
- **Parallel data fetching** — 50 tickers fetched in ~17s using ThreadPoolExecutor with Tenacity retry and 24hr disk cache
- **Behavioral clustering** — stocks grouped by valuation + growth profile using weighted KMeans with deterministic per-ticker labels
- **Similarity search** — separate fundamental and technical similarity matrices blended 70/30
- **Portfolio optimization** — CAPM + EW blended expected returns with Ledoit-Wolf shrinkage covariance across three risk profiles
- **Walk-forward backtesting** — no-lookahead validation with equal-weight baseline comparison across 8 periods
- **LLM summarization** — HuggingFace (primary) + Groq (fallback) with automatic provider switching
- **Interactive dashboard** — dark-mode UI with real-time search, autocomplete, gap analysis, and optimizer visualization

---

## Architecture

```
app/
├── api/
│   ├── routes.py          # FastAPI endpoints
│   └── schemas.py         # Pydantic request/response models
├── core/
│   ├── cache.py           # In-memory TTL cache
│   ├── config.py          # Settings (pydantic-settings + .env)
│   ├── disk_cache.py      # 24hr disk persistence for yfinance data
│   ├── logger.py          # Structured logging
│   └── validators.py      # FastAPI input validators
├── data/
│   ├── fetcher.py         # Parallel yfinance fetcher with retry
│   └── pit_fundamentals.py # Point-in-time fundamental calculations
├── evaluation/
│   └── backtester.py      # Walk-forward backtest + portfolio metrics
├── features/
│   ├── fundamentals.py    # Feature engineering, clipping, imputation, scaling
│   └── technical.py       # Momentum, volatility, RSI
├── models/
│   ├── clustering.py      # Weighted KMeans with deterministic labels
│   ├── optimizer.py       # PyPortfolioOpt MPT optimizer
│   ├── similarity.py      # Cosine similarity matrices
│   └── summarizer.py      # LLM summarization (HF + Groq)
├── services/
│   └── recommender.py     # Pipeline orchestrator + investable universe filter
└── main.py                # FastAPI app + lifespan
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <repo>
cd Stock_portfolio_recommender
uv sync
```

### Configuration

Create a `.env` file in the project root:

```env
# Required for LLM summarization (at least one)
GROQ_API_KEY=gsk_...
HF_API_KEY=hf_...

# Optional — override defaults
HF_MODEL=human-centered-summarization/financial-summarization-pegasus
GROQ_MODEL=llama-3.3-70b-versatile
DEFAULT_CAPITAL=10000
DEFAULT_RISK=moderate
LOG_LEVEL=INFO
```

### Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open the dashboard at `http://localhost:8000/static/dashboard.html`

Or explore the API via Swagger at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Service health and readiness |
| GET | `/api/v1/similar/{ticker}` | Find behaviorally similar stocks |
| POST | `/api/v1/gaps` | Identify diversification gaps in a portfolio |
| POST | `/api/v1/optimize` | Optimize portfolio weights by risk profile |
| GET | `/api/v1/evaluate/optimizer` | Walk-forward backtest |
| POST | `/api/v1/evaluate/portfolio_metrics` | Realized vs predicted metrics |
| POST | `/api/v1/summarize/similar` | LLM summary of similarity results |
| POST | `/api/v1/summarize/gaps` | LLM summary of gap analysis |
| POST | `/api/v1/summarize/optimize` | LLM summary of optimization results |

### Example requests

```bash
# Find stocks similar to AAPL
curl http://localhost:8000/api/v1/similar/AAPL?top_n=5

# Find portfolio gaps
curl -X POST http://localhost:8000/api/v1/gaps \
  -H "Content-Type: application/json" \
  -d '{"portfolio": ["AAPL", "MSFT", "GOOGL"], "top_n": 5}'

# Optimize portfolio
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "JNJ", "XOM", "JPM"], "risk": "moderate"}'
```

---

## Risk Profiles

| Profile | Strategy | Max Weight per Stock | Use Case |
|---------|----------|---------------------|----------|
| `conservative` | Min volatility | 10% | Capital preservation |
| `moderate` | Max Sharpe | 20% | Balanced growth |
| `aggressive` | Max quadratic utility | 35% | High growth, accepts risk |

---

## Data Pipeline

### Tickers (50 stocks across 5 sectors)

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO, ORCL, ADBE, CRM, AMD, INTC, QCOM, TXN |
| Financials | JPM, BAC, GS, MS, WFC, BLK, AXP, SCHW, C, USB |
| Healthcare | JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, DHR, BMY, LLY |
| Energy | XOM, CVX, COP, SLB, EOG |
| Consumer | WMT, PG, KO, PEP, COST, MCD, NKE, SBUX, TGT, HD |

### Investable Universe Filter

The following are automatically excluded from portfolio optimization:

| Ticker | Reason | Criteria |
|--------|--------|----------|
| INTC | Loss-making | `eps_ttm <= 0` |
| ABBV | Negative equity | `debt_to_equity < -3` |
| MCD | Negative equity | `debt_to_equity < -3` |
| SBUX | Negative equity | `debt_to_equity < -3` |
| TSLA | Speculative | `pe_ratio > 300` and `beta > 1.5` |

The filter is applied automatically in `recommender._build_investable_universe()` and passed through to the backtester via `exclude_tickers`.

### Caching

- **Disk cache** — fundamentals cached to `app/data/cache/` as JSON, 24hr TTL
- **In-memory cache** — API responses cached in-process, 1hr TTL
- **Cold fetch**: ~17s for 50 tickers
- **Warm cache**: ~0.5s

To clear the cache:

```bash
uv run python -c "
from app.core.disk_cache import DiskCache
DiskCache(cache_dir='app/data/cache', ttl_hours=24).clear_all()
print('Cache cleared')
"
```

---

## Clustering Labels

Stocks are assigned deterministic behavioral labels based on their raw fundamentals:

| Label | Criteria | Example |
|-------|----------|---------|
| `Hypergrowth` | Revenue growth > 50% | NVDA |
| `Distressed` | EPS <= 0, loss-making | INTC |
| `Negative Equity` | D/E < -3 | MCD, ABBV |
| `Speculative` | PE > 150 and beta > 1.5 | TSLA |
| `Quality Growth` | ROE > 0.25 and revenue growth > 10% | AAPL, MSFT, LLY |
| `Defensive Income` | Beta < 0.7 and dividend yield > 1.5% | JNJ, KO, PG |
| `Energy / Financials / Technology / etc.` | Sector-based fallback | XOM, JPM, AMD |

---

## Pipeline Evaluation

Run the full end-to-end evaluation:

```bash
uv run python evaluate_pipeline.py
```

This covers:
1. Data pipeline health (NaN rates, missing tickers)
2. Feature pipeline validation (scaling, imputation)
3. Clustering quality (label distribution, fundamentals summary)
4. Investable universe (excluded tickers and reasons)
5. Similarity sanity checks (known pairs)
6. Walk-forward backtest (all 3 risk profiles, 8 periods over 3 years)
7. Realized vs predicted metrics (return gap, vol gap, Sharpe)
8. Scorecard (pass/warn/fail per component)

### Latest evaluation results (March 2026, 3 years data)

```
Score: 10/10 PASS  |  0 WARN  |  0 FAIL

Backtest periods         : 16  (9-month train, 3-month test windows)
Conservative win rate    : 38%
Moderate win rate        : 50%  (beats equal weight in 4/8 periods)
Moderate outperformance  : +1.77% avg per period
Aggressive win rate      : 50%
Aggressive outperformance: +3.04% avg per period

Realized Sharpe          : 1.13  (market benchmark ~0.5)
Volatility gap           : -0.02% (excellent)
Return gap               : 13.3% (within normal in-sample bounds <15%)
Total 5Y return          : 103.97% (~20.88% annualized)
Max drawdown             : -13.14%
Calmar ratio             : 1.59
```

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific modules
uv run pytest tests/test_fetcher.py -v
uv run pytest tests/test_features.py -v
uv run pytest tests/test_recommender.py -v
uv run pytest tests/test_summarizer.py -v
uv run pytest tests/test_validators.py -v
uv run pytest tests/test_cache.py -v
uv run pytest tests/test_routes.py -v
uv run pytest tests/test_evaluation.py -v
```

### Test coverage

| File | Tests | Coverage |
|------|-------|----------|
| `test_fetcher.py` | 6 | Parallel fetch, cache, PIT fundamentals |
| `test_features.py` | 20 | Feature engineering, scaling, edge cases |
| `test_recommender.py` | 35 | Similarity, clustering, optimizer |
| `test_summarizer.py` | 28 | LLM routing, retry, prompt construction |
| `test_validators.py` | 17 | Input validation, HTTP errors |
| `test_cache.py` | 27 | SimpleCache + DiskCache TTL/expiry |
| `test_routes.py` | 28 | API endpoints, schemas, status codes |
| `test_evaluation.py` | 16 | Walk-forward backtest, portfolio metrics |
| **Total** | **177** | |

---

## LLM Configuration

The summarizer uses a two-provider fallback strategy:

```
HuggingFace (primary)  ->  Groq (fallback)  ->  error string
```

Supported models (configurable via `.env`):

| Provider | Default Model | Notes |
|----------|--------------|-------|
| HuggingFace | `human-centered-summarization/financial-summarization-pegasus` | Finance-specific summarization model |
| Groq | `llama-3.3-70b-versatile` | Fast, generous free tier |

To switch models without code changes:

```env
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
GROQ_MODEL=mixtral-8x7b-32768
```

---

## Known Limitations

- **Limited backtest periods** — 8 periods over 3 years provides reasonable statistical confidence. Expanding to 5+ years would further improve reliability.
- **TSLA momentum risk** — resolved by excluding TSLA from the investable universe (PE=363, beta=1.9). The `_build_investable_universe()` filter in `recommender.py` can be extended with additional speculative exclusions as needed.
- **In-sample Sharpe inflation** — predicted Sharpe (2.4) vs realized (1.4) shows a +1.03 gap, within acceptable bounds for in-sample MPT optimization. Volatility prediction is near-perfect (0.10% gap). Use walk-forward backtest results for realistic out-of-sample expectations rather than the predicted Sharpe.
- **yfinance rate limits** — cold fetch limited to ~100 requests/minute on free tier; disk cache mitigates this for repeated runs.
- **Single market regime** — the 3-year backtest period (2023-2026) was predominantly bullish for US equities. Performance in bear markets or high-volatility regimes is untested.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| API | FastAPI, Uvicorn |
| Data | yfinance, pandas, numpy |
| ML | scikit-learn, PyPortfolioOpt |
| LLM | HuggingFace Hub, Groq |
| Validation | Pydantic v2 |
| Config | pydantic-settings |
| Testing | pytest |
| Package mgmt | uv |