# StockIQ: Portfolio Recommendation Engine

A modular, API-first stock recommendation system combining content-based filtering, portfolio gap analysis, Modern Portfolio Theory optimization, and LLM-powered summaries — with a live dashboard UI.

---

## Quick Start

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/stock_recommender.git
cd stock_recommender

# Create virtual environment (Windows)
python -m venv .stockenv
.stockenv\Scripts\activate

# macOS/Linux
source .stockenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

```env
APP_ENV=development
APP_PORT=8000
LOG_LEVEL=INFO
DEFAULT_CAPITAL=10000
DEFAULT_RISK=moderate
GROQ_API_KEY=your_groq_api_key_here   # free at console.groq.com
```

### Run the API

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Wait for:
```
INFO | Recommender ready
```

Then open:
```
http://localhost:8000/docs            # Swagger UI
static/dashboard.html                 # Visual dashboard (open in browser)
```

---

## Features

**Stock Similarity Search**
Find stocks most similar to any given ticker using cosine similarity across 12 fundamental and technical features. Results include sector, beta, momentum, and volatility.

**Portfolio Gap Analysis**
Given an existing portfolio, identify low-correlation stocks across underrepresented sectors to improve diversification.

**Portfolio Optimizer**
Optimize portfolio weights using Modern Portfolio Theory via PyPortfolioOpt. Supports conservative, moderate, and aggressive risk profiles with automatic fallback when optimization fails.

**LLM-Powered Summaries**
Each result panel generates a plain-English analyst summary via Groq (Llama 3). Summaries are contextual — they reference actual numbers, sector concentrations, and risk tradeoffs from your results.

**Walk-Forward Backtesting**
Evaluate optimizer performance across rolling time windows with no lookahead bias. Compare optimized weights against an equal-weight baseline and measure predicted vs realized metrics.

**Live Dashboard**
Single-page HTML dashboard with real-time charts, weight visualizations, tag-based ticker input, and summarize buttons for all three panels.

---

## Project Structure

```
stock_recommender/
│
├── app/
│   ├── main.py                       # FastAPI entry, middleware, error handling
│   ├── api/
│   │   ├── routes.py                 # All API endpoints with response models
│   │   └── schemas.py                # Pydantic request/response schemas
│   ├── core/
│   │   ├── config.py                 # pydantic-settings with .env
│   │   ├── logger.py                 # Structured logging
│   │   ├── cache.py                  # In-memory TTL cache
│   │   └── validators.py             # Input validation helpers
│   ├── data/
│   │   └── fetcher.py                # yfinance data fetching
│   ├── features/
│   │   ├── technical.py              # RSI, momentum, volatility
│   │   └── fundamental.py            # PE, beta, scaling, imputation
│   ├── models/
│   │   ├── similarity.py             # Cosine similarity engine
│   │   ├── clustering.py             # KMeans stock clustering
│   │   └── optimizer.py              # PyPortfolioOpt wrapper with fallback
│   ├── services/
│   │   ├── recommender.py            # Master service, singleton, caching
│   │   └── summarizer.py             # Groq LLM summary generation
│   └── evaluation/
│       └── backtester.py             # Walk-forward backtesting + metrics
│
├── static/
│   └── dashboard.html                # Frontend UI
│
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── test_fetcher.py               # Data fetching tests
│   ├── test_features.py              # Feature engineering tests
│   ├── test_recommender.py           # Similarity, clustering, optimizer tests
│   ├── test_evaluation.py            # Backtester tests
│   └── test_summarizer.py            # LLM summarizer tests (mocked)
│
├── .github/
│   └── workflows/
│       └── test.yml                  # GitHub Actions CI
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```
```json
{
  "status": "ok",
  "ready": true,
  "ticker_count": 18,
  "uptime_seconds": 142.3
}
```

### Similar Stocks
```bash
GET /api/v1/similar/{ticker}?top_n=5
```
```json
[
  {
    "ticker": "NVDA",
    "sector": "Technology",
    "beta": 2.31,
    "momentum_6m": 0.084,
    "volatility": 0.496,
    "similarity": 0.675
  }
]
```

### Portfolio Gap Analysis
```bash
POST /api/v1/gaps
```
```json
{ "portfolio": ["AAPL", "MSFT"], "top_n": 5 }
```
```json
[
  { "ticker": "JNJ", "sector": "Healthcare", "correlation": -0.052 },
  { "ticker": "KO",  "sector": "Consumer",   "correlation":  0.010 }
]
```

### Portfolio Optimizer
```bash
POST /api/v1/optimize
```
```json
{ "tickers": ["AAPL", "MSFT", "JNJ", "XOM"], "risk": "moderate" }
```
```json
{
  "weights":         { "AAPL": 0.031, "MSFT": 0.284, "JNJ": 0.469, "XOM": 0.216 },
  "expected_return": 0.1823,
  "volatility":      0.1209,
  "sharpe_ratio":    1.0939
}
```

Risk options: `conservative` · `moderate` · `aggressive`

### LLM Summaries
```bash
POST /api/v1/summarize/similar
POST /api/v1/summarize/gaps
POST /api/v1/summarize/optimize
```
```json
{ "summary": "AAPL's closest peers are concentrated in Technology and Communication Services..." }
```

### Backtesting
```bash
GET /api/v1/evaluate/optimizer?tickers=AAPL,MSFT,JNJ,XOM&risk=moderate
```
```json
{
  "summary": {
    "periods_tested": 4,
    "avg_optimized_return": 0.0412,
    "avg_equal_weight_return": 0.0381,
    "avg_outperformance": 0.0031,
    "win_rate_vs_equal": 0.75,
    "best_period": "2022-04-01",
    "worst_period": "2022-10-01"
  },
  "tradeoffs": {
    "method": "walk_forward",
    "note": "Walk-forward prevents lookahead bias. Simple hold-out would overestimate performance."
  }
}
```

```bash
POST /api/v1/evaluate/portfolio_metrics
```
```json
{
  "predicted": { "expected_return": 0.18, "volatility": 0.12, "sharpe_ratio": 1.09 },
  "realized":  { "realized_annual_return": 0.14, "realized_volatility": 0.13, "realized_sharpe": 0.69 },
  "gap": {
    "return_gap": 0.04,
    "note": "Positive gap = optimizer was optimistic. Expected for in-sample prediction."
  }
}
```

### Cache Stats
```bash
GET /api/v1/cache/stats
```
```json
{ "hits": 12, "misses": 3, "hit_rate": 0.8, "size": 3 }
```

---

## How It Works

### Data Pipeline

```
yfinance (live data)
      │
      ├── Price history (2y) ──► Technical features
      │                          (RSI, momentum_3m, momentum_6m, volatility)
      │
      └── Fundamentals ────────► Fundamental features
                                  (PE, PB, ROE, debt/equity, revenue growth,
                                   dividend yield, beta, market cap)
                │
                └──── Merge + Median Imputation + StandardScale
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              KMeans(5)     Cosine      Efficient
              Clusters    Similarity   Frontier
                          Matrix       (MPT)
```

### Recommendation Logic

**Similarity** — cosine similarity across 12 scaled features. Scale-invariant by design — a stock with high PE and high beta isn't penalized relative to one that's just high PE.

**Gap Analysis** — computes Pearson correlation of each candidate's daily returns against the portfolio's average return. Lower correlation = better diversifier. Sector info is attached from the fundamental data so results are always labeled.

**Optimization** — PyPortfolioOpt Efficient Frontier with three strategies and automatic fallback chain:
- `conservative` → minimize volatility
- `moderate` → maximize Sharpe ratio
- `aggressive` → maximize quadratic utility
- fallback 1 → min volatility if primary strategy is infeasible
- fallback 2 → relaxed weight bounds (0, 1) if still infeasible

**Summarization** — results are serialized and sent to Groq (Llama 3 8B) with a financial analyst system prompt. Temperature is kept low (0.4) to reduce hallucination risk.

---

## Observability

Every request is instrumented automatically.

**Request ID** — every response includes `X-Request-ID` header for tracing.

**Response timing** — every response includes `X-Response-Time` in milliseconds.

**Structured logs** — all log lines include method, path, status code, and latency:
```
INFO | request_completed | GET /api/v1/similar/AAPL | 200 | 4.2ms
```

**Global error handler** — all unhandled exceptions return a consistent JSON shape instead of a raw 500:
```json
{ "error": "Internal server error", "detail": "...", "path": "/api/v1/..." }
```

**Cache stats** — hit rate, size, and miss count available at `/api/v1/cache/stats`.

---

## Caching

Results for `similar`, `gaps`, and `optimize` are cached in-memory with a 1-hour TTL.

```
GET /api/v1/similar/AAPL   →  miss  →  compute  →  cache
GET /api/v1/similar/AAPL   →  hit   →  return immediately
```

Cache is invalidated automatically when the model is rebuilt.

**Tradeoff — in-memory vs Redis:** In-memory cache is zero-dependency and works perfectly for a single-worker deployment. With multiple uvicorn workers, each process has its own cache so the same request may be computed multiple times across workers. Redis solves this but adds operational complexity. For a single-server deployment, in-memory is the right tradeoff.

---

## Evaluation & Backtesting

### Walk-Forward Validation

The backtester uses a rolling window approach to avoid lookahead bias:

```
|── 12 months train ──|── 3 months test ──|
                      |── 12 months train ──|── 3 months test ──|
                                            |── 12 months train ──| ...
```

At each step the optimizer is trained on historical data only, then evaluated on the next unseen period. This is the correct method for time-series — a simple train/test split would leak future information and overestimate performance.

### Predicted vs Realized Metrics

The `/evaluate/portfolio_metrics` endpoint compares what the optimizer predicted against what actually happened historically:

| Metric | Predicted | Realized | Gap |
|---|---|---|---|
| Annual Return | 18.2% | 14.1% | +4.1% |
| Volatility | 12.1% | 13.4% | -1.3% |
| Sharpe Ratio | 1.09 | 0.69 | +0.40 |

A positive return gap (optimizer was optimistic) is expected — the optimizer is trained on the same data it predicts over, so it overfits to historical patterns. Walk-forward validation corrects for this by measuring on unseen periods.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific module
python -m pytest tests/test_evaluation.py -v
python -m pytest tests/test_summarizer.py -v
```

Expected output:
```
63 passed in ~9s
```

Test coverage by module:

| Module | Tests | What's covered |
|---|---|---|
| `test_fetcher.py` | 4 | yfinance mocking, missing fields, error handling |
| `test_features.py` | 8 | RSI, momentum, scaling, imputation, merging |
| `test_recommender.py` | 12 | Similarity matrix, clustering, optimizer, edge cases |
| `test_evaluation.py` | 16 | Walk-forward correctness, metrics, math validation |
| `test_summarizer.py` | 20 | Groq mocked, prompt content, error handling |

All Groq API calls in `test_summarizer.py` are mocked — tests run instantly with no API key or network needed.

---

## CI/CD

GitHub Actions runs the full test suite on every push to `main` or `dev` and on every pull request. Pipeline fails if coverage drops below 70%.

```yaml
# .github/workflows/test.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ -v --cov=app --cov-fail-under=70
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| API framework | FastAPI + Uvicorn |
| Data | yfinance |
| Feature engineering | pandas, numpy, scikit-learn |
| Optimization | PyPortfolioOpt |
| LLM summarization | Groq (Llama 3 8B) |
| Config & validation | pydantic-settings, pydantic v2 |
| Caching | In-memory TTL (SimpleCache) |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |

---

## Key Design Decisions & Tradeoffs

**Cosine similarity vs neural embeddings** — cosine on scaled tabular features is interpretable, fast, and requires no training data. A learned embedding (autoencoder or contrastive learning) would capture non-linear relationships but needs periodic retraining as market regimes shift and is harder to explain to non-technical stakeholders.

**In-memory model vs persistent store** — the model rebuilds on every startup (~30s). Simple and always fresh, but loses state on crash. Serializing with `joblib.dump()` would give instant startup at the cost of potential staleness if market data changes between restarts.

**MPT optimizer vs ML-based allocation** — MPT is theoretically grounded, interpretable, and well-understood by practitioners. It assumes normally distributed returns and relies on historical covariance, which breaks during market regime changes. RL-based or Black-Litterman models handle non-stationarity better but are significantly harder to explain.

**Walk-forward vs hold-out evaluation** — simple train/test split leaks time information. Walk-forward is the correct method for financial time series but requires more data — at least 15 months for one meaningful test cycle with default settings (12m train + 3m test).

**In-memory cache vs Redis** — right answer depends on deployment topology. One worker: in-memory is fine. Multiple workers: Redis is required. Chose in-memory to keep the dependency surface small.

**Groq vs OpenAI for summaries** — Groq's free tier is fast enough for interactive use (< 1s response) and Llama 3 8B produces coherent financial summaries. OpenAI GPT-4 would produce higher quality output but at cost. Summaries are supplementary, not critical — Groq is the right tradeoff.

---

## Limitations

- Stock universe is fixed at startup — adding new tickers requires a restart and full model rebuild
- No persistent storage — model rebuilds from scratch each startup (~30 seconds)
- Optimizer can fail with fewer than 3 tickers or in extreme low-variance conditions — automatic fallback to min-volatility is applied
- yfinance is an unofficial Yahoo Finance API — subject to rate limits, schema changes, and occasional data gaps
- LLM summaries use Groq's free tier — subject to rate limits under heavy use
- Walk-forward backtesting requires at least 15 months of price history to produce meaningful results

---

