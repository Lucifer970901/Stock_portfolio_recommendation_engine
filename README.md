# Stock Portfolio Recommendation Engine

An end-to-end stock portfolio recommendation engine built with FastAPI, PyPortfolioOpt, and LLM-powered summarization. The system fetches real-time fundamentals, clusters stocks by behavioral similarity, optimizes portfolios using Modern Portfolio Theory, and explains results in plain English.

---

## Features

- **Point-in-time fundamentals** — PE ratio, EPS TTM, revenue growth, D/E ratio calculated from quarterly reports with 45-day reporting lag to prevent lookahead bias
- **Parallel data fetching** — 50 tickers fetched in ~20s using ThreadPoolExecutor with Tenacity retry and 24hr disk cache
- **Behavioral clustering** — stocks grouped by valuation + growth profile using weighted KMeans with deterministic per-ticker labels
- **Similarity search** — separate fundamental and technical similarity matrices blended 70/30
- **Portfolio optimization** — CAPM + EW blended expected returns with Ledoit-Wolf shrinkage covariance across three risk profiles
- **Walk-forward backtesting** — no-lookahead validation with equal-weight baseline comparison across 16 periods over 5 years
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
- **Cold fetch**: ~20s for 50 tickers (5Y data)
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
6. Walk-forward backtest (all 3 risk profiles, 16 periods over 5 years)
7. Realized vs predicted metrics (return gap, vol gap, Sharpe)
8. Scorecard (pass/warn/fail per component)

### Latest evaluation results (March 2026, 5 years data)

```
Score: 10/10 PASS  |  0 WARN  |  0 FAIL

Data                     : 1256 trading days (5 years)
Backtest periods         : 16  (9-month train, 3-month test windows)

Conservative win rate    : 31%  (expected — min vol underperforms in bull markets)
Conservative outperform  : -1.82% avg (capital preservation, not outperformance)

Moderate win rate        : 50%  (beats equal weight in 8/16 periods)
Moderate outperformance  : +1.77% avg per period

Aggressive win rate      : 50%
Aggressive outperformance: +3.04% avg per period

Realized Sharpe          : 1.13  (market benchmark ~0.5)
Volatility gap           : -0.02% (near-perfect prediction)
Return gap               : 13.3% (within normal in-sample bounds <15%)
Total 5Y return          : 103.97% (~14.9% annualized)
Max drawdown             : -13.14%
Calmar ratio             : 1.59
Evaluation time          : 33.9s
```

### Notable backtest periods

| Period | Profile | Result | Driver |
|--------|---------|--------|--------|
| Mar–Jun 2023 | Aggressive | +45.2% vs +9.7% eq | META rally (+35.5% out) |
| Mar–Jun 2023 | Moderate | +28.2% vs +9.7% eq | META rally (+18.4% out) |
| Sep–Dec 2024 | Aggressive | +13.9% vs +0.4% eq | NVDA rally (+13.5% out) |
| Mar–Jun 2022 | All profiles | Beat eq weight | Energy rotation, CVX top pick |
| Dec 2021 | Aggressive | -8.4% vs +0.5% eq | NVDA crash (-26.7% mdd) |

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

- **Conservative profile underperforms in bull markets** — by design. Min volatility concentrates in low-beta defensives (JNJ, PG) which lag in strong bull runs. Win rate of 31% over 5 years reflects this tradeoff; it is intended for capital preservation, not outperformance.
- **TSLA momentum risk** — resolved by excluding TSLA from the investable universe (PE=363, beta=1.9). The `_build_investable_universe()` filter in `recommender.py` can be extended with additional speculative exclusions as needed.
- **In-sample Sharpe inflation** — predicted Sharpe (2.1) vs realized (1.1) shows a +0.94 gap, within acceptable bounds for in-sample MPT optimization. Volatility prediction is near-perfect (-0.02% gap). Use walk-forward backtest results for realistic out-of-sample expectations rather than the predicted Sharpe.
- **Concentration risk in aggressive profile** — worst drawdown of -26.7% (Dec 2021, NVDA) shows the aggressive profile can suffer severe single-quarter losses. Position sizing and stop-loss logic are not implemented.
- **yfinance rate limits** — cold fetch ~20s for 50 tickers over 5 years; disk cache mitigates this for repeated runs.


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

---

## Design Tradeoffs

### Data & Fundamentals

* **Point-in-time vs latest fundamentals** — the pipeline applies a 45-day reporting lag to quarterly data to prevent lookahead bias. This means the fundamentals used for clustering and optimization reflect what was actually known at the time, not restated values. The tradeoff is that very recent earnings surprises (within 45 days) won't be reflected until the next cache refresh.

* **yfinance vs paid data provider** — yfinance is free and sufficient for a 50-ticker universe but has rate limits, occasional data gaps, and no survivorship bias protection. A paid provider (Polygon, Tiingo, Bloomberg) would give cleaner data and point-in-time index membership, at significant cost.

* **Disk cache TTL of 24 hours** — fundamentals are cached for 24 hours to avoid hammering the yfinance API. This means intraday fundamental changes (rare) won't be reflected until the next day. The tradeoff between freshness and reliability favors the cache for a daily-rebalanced system.

### Feature Engineering

* **Clipping outliers vs removing them** — extreme values (PE > 200, D/E > 10) are clipped rather than dropped. This preserves all 50 tickers in the universe while preventing a single outlier from distorting the scaled feature space. The tradeoff is that clipped values lose precision at the extremes.

* **Median imputation vs model-based imputation** — missing fundamentals are filled with the median of the column. This is simple, interpretable, and non-leaky. The tradeoff is it assumes the missing value is typical, which may not hold for distressed or unusual stocks like INTC.

* **Weighted KMeans (fundamentals 2x, technical 1x, engineered 0.5x)** — fundamental features are upweighted because they reflect durable business characteristics, while technical features reflect short-term momentum. The tradeoff is the optimizer may underweight momentum signals that are genuinely predictive over 3-month horizons.

### Clustering

* **Deterministic per-ticker labels vs centroid-based labels**  — cluster labels are assigned from each ticker's raw fundamentals directly, not from KMeans centroids. This makes labels stable and interpretable across runs. The tradeoff is that the label doesn't capture the relative position within a cluster — two Quality Growth stocks may be very different from each other.

* **Fixed n_clusters=8** — chosen from elbow analysis on the 50-ticker universe. With only 50 stocks, more clusters produce singletons (NVDA, TSLA, INTC, SLB are natural outliers). The tradeoff is that 4 singleton clusters exist, which is cosmetically unsatisfying but fundamentally correct — these stocks are genuinely outliers.

### Portfolio Optimization

* **CAPM + EW blended returns (70/30)** — pure CAPM anchors all expected returns to Rf + beta * market_premium, making stocks indistinguishable when beta is similar. Blending with exponentially weighted historical returns (EW) adds cross-sectional differentiation. The tradeoff is EW can overfit to recent momentum, which caused TSLA and AVGO overweighting in some periods.

* **Ledoit-Wolf shrinkage covariance** — shrinks the sample covariance matrix toward a structured estimator, reducing estimation error with limited data. The tradeoff vs sample covariance is a slight bias toward equal correlations, but this is almost always preferable with fewer than 500 observations per ticker.

* **Three-stage optimization fallback** — the optimizer tries max_sharpe (or min_volatility / max_quadratic_utility) first, then falls back to min_volatility with the same bounds, then relaxes bounds entirely. This prevents hard failures at the cost of occasionally producing a more conservative portfolio than requested.

* **Investable universe filter** — stocks are excluded based on eps_ttm <= 0, debt_to_equity < -3, or pe_ratio > 300 + beta > 1.5. This is a rules-based filter, not a predictive model. The tradeoff is it may exclude stocks that are recovering (e.g. INTC could return to profitability) and includes stocks that may deteriorate.

### Backtesting

* **Walk-forward vs simple train/test split** — walk-forward retraining at each period prevents lookahead bias and gives a realistic picture of out-of-sample performance. The tradeoff is it requires more data (minimum ~15 months for even 2 periods) and is slower to compute than a single split.

* **Equal weight as baseline** — the benchmark is a naive equal-weight portfolio of the same ticker universe. This is a low bar — a more rigorous benchmark would be SPY or a factor-adjusted index. The moderate optimizer beating equal weight 50% of the time over 16 periods means it adds value, but not dramatically so.

* **9-month training window** — shorter than the 12-month default to maximize the number of test periods from 5 years of data (16 periods vs 12). The tradeoff is the optimizer trains on slightly less data per window, which can increase variance in weight estimates.

---