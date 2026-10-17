# Equity Portfolio Analysis

A reproducible, single-file Python toolkit for empirical asset-pricing and portfolio analysis on a
monthly panel of stocks: from descriptive statistics to mean-variance optimisation, CAPM/GRS testing,
out-of-sample backtesting and momentum sorts.

> Originally written for the *Intelligent Methods for Financial Data Analysis* course (2026) and
> refactored into a tested, documented tool. Author: Zisis Akrivos

## What it does

| Step | Analysis | Method |
|------|----------|--------|
| 1 | Descriptive statistics | Mean, volatility, skewness, kurtosis, Sharpe, covariance and correlation |
| 2 | Tangency portfolio + CAPM | Closed-form max-Sharpe weights, per-stock CAPM regressions, joint **GRS test** (exact F p-value) |
| 3 | Out-of-sample backtest | Rolling 5-year window, annual rebalancing, drifting weights; unconstrained vs long-only tangency vs 1/N |
| 4 | Momentum (12-1) | 10 decile portfolios, CAPM alphas, winners-minus-losers (WML), Spearman monotonicity test |

## Quick start

```bash
git clone https://github.com/zisisakrivos05-ops/equity-portfolio-analysis.git
cd equity-portfolio-analysis
python equity_portfolio_analysis.py --demo        # synthetic data, no private file needed
python equity_portfolio_analysis.py --init repo--author"Zisis Akrivos" --repo-url https://github.com/USERNAME/equity-portfolio-analysis.git  # built-in unit tests
python equity_portfolio_analysis.py --data data/your_file.xlsx --rf 0.03 --window-years 5
```

Missing packages are installed automatically on first run (or `pip install -r requirements.txt`).
Outputs (CSV tables, `summary.json`, PNG figures) are written to `results/`.

### Input format
Excel workbook with sheets `STOCKS`, `market_cap` and `returns` (dates in the first column, one column
per ticker, returns as decimals, e.g. `0.05` = 5%; percentages are detected and converted).

## Methodological notes

- **No look-ahead**: lagged market caps for the market proxy, momentum uses only past returns (month t-1
  skipped), OOS weights use data up to the rebalance date. Covered by the self-tests.
- **Missing data is not zero-filled** for estimation or ranking; the GRS test uses a balanced sub-panel.
- **Risk-free rate** is a constant 3% p.a. (no reliable series in the data); change with `--rf`.
- **Unconstrained mean-variance is unstable** with 50 assets and 60 observations (extreme leverage,
  large out-of-sample losses). That is why a long-only variant and a 1/N benchmark are reported next to
  it. A portfolio losing 100% is treated as wiped out. Optional cap: `--max-leverage`.
- Standard errors are classical OLS (no Newey-West); a 50-stock sub-universe is illustrative, not a
  statement about the whole market.

## License
MIT

## Demo output (synthetic data)

The figures below come from `--demo` (randomly generated data), **not** from real market data.

![Out-of-sample performance](docs/q4_oos_cumulative.png)
![Security Market Line](docs/q2_sml.png)
![Momentum deciles](docs/q5_momentum.png)

In this demo the unconstrained tangency portfolio is wiped out (extreme leverage from estimation error),
while the long-only portfolio and the 1/N benchmark behave sensibly.
