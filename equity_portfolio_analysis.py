#!/usr/bin/env python3
# Copyright (c) 2026 Zisis. Released under the MIT License
# (the LICENSE file is written by:  python equity_portfolio_analysis.py --init-repo).
"""
Equity Portfolio Analysis
=========================
Author: Zisis Akrivos
Empirical asset-pricing toolkit for a monthly panel of stocks (single file, CLI).

Analyses
--------
1. Descriptive statistics     mean, volatility, skew, kurtosis, Sharpe, covariance, correlation
2. Tangency portfolio + CAPM  max-Sharpe weights, per-stock CAPM, joint GRS test (exact F p-value)
3. Out-of-sample backtest     rolling window, annual rebalancing, weights drift between rebalances;
                              unconstrained tangency vs long-only tangency vs 1/N
4. Momentum (12-1)            10 decile portfolios, CAPM alphas, winners-minus-losers (WML),
                              Spearman monotonicity test

Input
-----
Excel workbook with sheets  STOCKS (metadata), market_cap, returns
(first column = dates, one column per ticker, returns as decimals: 0.05 = 5%).

Usage  (missing packages are installed automatically on first run)
-----
    python equity_portfolio_analysis.py --data data/your_file.xlsx
    python equity_portfolio_analysis.py --demo          # synthetic data, no private file needed
    python equity_portfolio_analysis.py --selftest      # run built-in sanity tests
    python equity_portfolio_analysis.py --init-repo   # write README, LICENSE, .gitignore...

Methodological notes
--------------------
* No look-ahead: market weights use lagged market caps; momentum uses only past returns
  (month t-1 skipped); OOS weights are estimated on data up to the rebalance date.
* Missing data is NOT zero-filled for estimation: a stock needs a complete window to enter an
  optimisation or a momentum ranking; the GRS test uses a balanced sub-panel.
* Risk-free rate: constant (default 3% p.a.), since no reliable series is available.
* The unconstrained tangency portfolio is numerically unstable (50 assets, 60 monthly observations);
  expect extreme leverage and large losses out of sample (estimation error), which is why
  a long-only variant and a 1/N benchmark are reported alongside. An optional gross-leverage cap
  (--max-leverage, default: none) falls back to 1/N when breached. A portfolio that loses 100%
  is treated as wiped out (wealth cannot go negative).
* Standard errors are classical OLS (no Newey-West).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------------ dependency bootstrap
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
            "matplotlib": "matplotlib", "openpyxl": "openpyxl"}      # import name -> pip name


def ensure_dependencies(required: dict[str, str] = REQUIRED) -> None:
    """Install any missing third-party package automatically (set EQPORT_NO_AUTOINSTALL=1 to disable)."""
    import importlib.util
    import os
    import subprocess

    missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    manual = f'"{sys.executable}" -m pip install {" ".join(missing)}'
    if os.environ.get("EQPORT_NO_AUTOINSTALL"):
        sys.exit(f"Missing packages: {', '.join(missing)}\nInstall them with:\n  {manual}")

    print(f"[setup] Missing packages: {', '.join(missing)}. Installing automatically (one-time)...")
    for extra in ([], ["--user"]):                      # second attempt: per-user install
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *extra, *missing]
        if subprocess.run(cmd).returncode == 0:
            importlib.invalidate_caches()
            if all(importlib.util.find_spec(m) for m in required):
                print("[setup] Done.\n")
                return
    sys.exit("\n[setup] Automatic installation failed (no internet, or a managed Python environment).\n"
             f"Install manually with:\n  {manual}\n"
             "If you see 'externally-managed-environment', create a virtual environment first:\n"
             "  python3 -m venv .venv && source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)\n"
             f"  pip install {' '.join(required.values())}")


ensure_dependencies()

import matplotlib
matplotlib.use("Agg")                       # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize

__version__ = "1.0.0"
log = logging.getLogger("eqport")

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#2b6cb0", "#c53030", "#2f855a", "#718096"


# =============================================================================== CONFIG
@dataclass(frozen=True)
class Config:
    n_stocks: int = 50            # keep the N largest stocks by average market cap
    min_obs: int = 36             # minimum observations for a stock-level model
    rf_annual: float = 0.03       # constant annual risk-free rate
    window_years: int = 5         # rolling estimation window (OOS)
    rebalance_month: int = 12     # rebalance every December
    mom_lookback: int = 12        # momentum signal history (months)
    mom_skip: int = 1             # most recent month(s) skipped
    n_deciles: int = 10
    max_gross_leverage: float | None = None   # optional cap on gross exposure of the unconstrained portfolio

    @property
    def rf_monthly(self) -> float:
        return (1.0 + self.rf_annual) ** (1.0 / 12.0) - 1.0


# =============================================================================== DATA
@dataclass
class Dataset:
    returns: pd.DataFrame
    mcap: pd.DataFrame
    info: pd.DataFrame


def _read_panel(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, index_col=0, engine="openpyxl")
    df.index = pd.to_datetime(df.index, errors="coerce", dayfirst=True)
    df = df[~df.index.isna()].apply(pd.to_numeric, errors="coerce")
    df.columns = [str(c).strip() for c in df.columns]
    return df.sort_index()


def _find_sheet(names: list[str], wanted: str) -> str:
    """Case/whitespace-insensitive lookup (the raw file has a sheet called ' returns')."""
    for n in names:
        if n.strip().lower() == wanted.lower():
            return n
    raise KeyError(f"Sheet '{wanted}' not found. Available sheets: {names}")


def load_dataset(path: str | Path, cfg: Config) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}\n"
                                "Use --data /path/to/file.xlsx or --demo for synthetic data.")
    names = pd.ExcelFile(path, engine="openpyxl").sheet_names
    returns = _read_panel(path, _find_sheet(names, "returns"))
    mcap = _read_panel(path, _find_sheet(names, "market_cap"))
    try:
        info = pd.read_excel(path, sheet_name=_find_sheet(names, "stocks"), engine="openpyxl")
        info.columns = [str(c).strip() for c in info.columns]
    except KeyError:
        info = pd.DataFrame()

    common = returns.columns.intersection(mcap.columns)
    if len(common) == 0:
        raise ValueError("No common tickers between 'returns' and 'market_cap' sheets.")
    returns, mcap = returns[common], mcap[common]

    if returns.shape[1] > cfg.n_stocks:
        keep = mcap.mean().nlargest(cfg.n_stocks).index
        returns, mcap = returns[keep], mcap[keep]
    returns = returns.dropna(how="all")
    mcap = mcap.reindex(returns.index)

    # sanity check on units: returns must be decimals, not percentages
    med_abs = returns.abs().stack().median()
    if med_abs > 0.5:
        log.warning("Median |return| = %.2f: values look like PERCENTAGES. "
                    "Dividing by 100 (use decimals in the file to avoid this).", med_abs)
        returns = returns / 100.0

    cov = returns.notna().mean()
    log.info("Universe: %d stocks, %d months (%s -> %s)", returns.shape[1], len(returns),
             returns.index.min().date(), returns.index.max().date())
    log.info("Coverage per stock: min=%.0f%% median=%.0f%% max=%.0f%%",
             cov.min() * 100, cov.median() * 100, cov.max() * 100)
    return Dataset(returns, mcap, info)


# =============================================================================== STATS
def descriptive_stats(R: pd.DataFrame, rf_m: float) -> pd.DataFrame:
    out = pd.DataFrame({"mean": R.mean(), "std": R.std(), "var": R.var(), "min": R.min(),
                        "max": R.max(), "skew": R.skew(), "kurt": R.kurt(), "n_obs": R.notna().sum()})
    out["ann_mean"] = out["mean"] * 12
    out["ann_std"] = out["std"] * np.sqrt(12)
    out["sharpe_ann"] = (out["mean"] - rf_m) / out["std"] * np.sqrt(12)
    return out.sort_values("mean", ascending=False)


def mean_pairwise_corr(corr: pd.DataFrame) -> float:
    return float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())


def market_proxy(R: pd.DataFrame, mcap: pd.DataFrame) -> pd.Series:
    """Value-weighted market using LAGGED market caps; weights renormalised over stocks with a return."""
    w = mcap.reindex(index=R.index, columns=R.columns).shift(1).where(R.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (w * R).sum(axis=1, min_count=1).rename("market")


def ols_capm(y: pd.Series, x: pd.Series, min_obs: int = 24) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    n = len(df)
    nan = {k: np.nan for k in ("alpha", "beta", "t_alpha", "t_beta", "r2")}
    if n < min_obs:
        return {**nan, "n_obs": n}
    yv, xv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones(n), xv])
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    e = yv - X @ b
    se = np.sqrt(np.diag(e @ e / (n - 2) * np.linalg.inv(X.T @ X)))
    sst = ((yv - yv.mean()) ** 2).sum()
    return {"alpha": b[0], "beta": b[1], "t_alpha": b[0] / se[0], "t_beta": b[1] / se[1],
            "r2": 1 - e @ e / sst if sst > 0 else np.nan, "n_obs": n}


def capm_table(R: pd.DataFrame, rf: pd.Series, mkt: pd.Series, min_obs: int) -> pd.DataFrame:
    mkt_ex = mkt - rf
    out = pd.DataFrame({c: ols_capm(R[c] - rf, mkt_ex, min_obs) for c in R.columns}).T
    out["alpha_ann_pct"] = out["alpha"] * 12 * 100
    return out


def grs_test(R: pd.DataFrame, rf: pd.Series, mkt: pd.Series) -> dict:
    """Gibbons-Ross-Shanken (1989): F = (T-N-1)/N * a'S^-1 a / (1+SR_m^2) ~ F(N, T-N-1) under H0."""
    panel = pd.concat([R.sub(rf, axis=0), (mkt - rf).rename("__mkt__")], axis=1)
    panel = panel.dropna(axis=1, thresh=int(0.999 * len(panel))).dropna(axis=0)
    assets = [c for c in panel.columns if c != "__mkt__"]
    T, N = len(panel), len(assets)
    if N == 0 or T <= N + 1:
        return {"F": np.nan, "p_value": np.nan, "N": N, "T": T}
    Y, x = panel[assets].to_numpy(), panel["__mkt__"].to_numpy()
    X = np.column_stack([np.ones(T), x])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    Sigma, a = E.T @ E / T, B[0]
    sr2 = x.mean() ** 2 / x.var()
    F = (T - N - 1) / N * float(a @ np.linalg.pinv(Sigma) @ a) / (1 + sr2)
    return {"F": F, "p_value": float(sps.f.sf(F, N, T - N - 1)), "N": N, "T": T}


# =============================================================================== OPTIMISATION
def tangency_weights(mu: pd.Series, cov: pd.DataFrame, rf: float, long_only: bool = False,
                     max_gross_leverage: float | None = None) -> pd.Series:
    """Max-Sharpe portfolio (weights sum to 1).

    Unconstrained solution: w ∝ Σ^-1 (μ - rf). Because it is normalised by the sum of raw weights,
    it explodes when that sum is close to zero; if the solution is degenerate or its gross exposure
    sum|w| exceeds `max_gross_leverage`, the function falls back to 1/N.
    """
    n = len(mu)
    ex, S = (mu - rf).to_numpy(), cov.to_numpy()
    equal = pd.Series(np.full(n, 1.0 / n), index=mu.index)
    if not long_only:
        raw = np.linalg.pinv(S) @ ex
        tot = raw.sum()
        if not np.isfinite(tot) or tot <= 1e-10:
            log.debug("Degenerate unconstrained tangency solution; using 1/N.")
            return equal
        w = pd.Series(raw / tot, index=mu.index)
        if max_gross_leverage is not None and w.abs().sum() > max_gross_leverage:
            log.debug("Gross leverage %.1fx > cap %.1fx; using 1/N.", w.abs().sum(), max_gross_leverage)
            return equal
        return w

    res = minimize(lambda w: -(w @ ex) / np.sqrt(max(w @ S @ w, 1e-16)), np.full(n, 1.0 / n),
                   method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success or not np.all(np.isfinite(res.x)):
        log.debug("Long-only optimiser did not converge (%s); using 1/N.", res.message)
        return equal
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=mu.index)


def portfolio_summary(r: pd.Series, rf_m: float) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    return {"mean_monthly": r.mean(), "std_monthly": r.std(), "ann_mean": r.mean() * 12,
            "ann_std": r.std() * np.sqrt(12), "sharpe_ann": (r.mean() - rf_m) / r.std() * np.sqrt(12),
            "min": r.min(), "max": r.max(), "skew": r.skew(), "kurt": r.kurt(),
            "max_drawdown": float((cum / cum.cummax() - 1).min()), "n_months": int(len(r))}


def rolling_oos(R: pd.DataFrame, cfg: Config, long_only: bool = False):
    """Rolling-window tangency portfolio; held (with drifting weights) until the next rebalance.
    Missing returns after the estimation date (delistings) are treated as 0%."""
    W, dates = cfg.window_years * 12, R.index
    rebal = [i for i, d in enumerate(dates) if d.month == cfg.rebalance_month and i >= W - 1]
    if not rebal:
        raise ValueError("Not enough history for the requested estimation window.")
    out, hist, ruined = [], {}, False
    for k, i in enumerate(rebal):
        if ruined:
            break
        win = R.iloc[i - W + 1: i + 1]
        win = win.loc[:, win.notna().all()]
        if win.shape[1] < 2:
            continue
        w = tangency_weights(win.mean(), win.cov(), cfg.rf_monthly, long_only, cfg.max_gross_leverage)
        hist[dates[i]] = w
        end = rebal[k + 1] if k + 1 < len(rebal) else len(dates) - 1
        cur = w.copy()
        for j in range(i + 1, end + 1):
            r = R.iloc[j][cur.index].fillna(0.0)
            pr = float(cur @ r)
            if 1 + pr <= 1e-6:                      # leveraged portfolio wiped out: wealth cannot go below 0
                log.warning("Unconstrained portfolio wiped out on %s (return <= -100%%); "
                            "all later months are set to 0%%.", dates[j].date())
                out.append((dates[j], -1.0))
                out.extend((dates[q], 0.0) for q in range(j + 1, len(dates)))
                ruined = True
                break
            out.append((dates[j], pr))
            cur = cur * (1 + r) / (1 + pr)
    name = "oos_long_only" if long_only else "oos_unconstrained"
    return pd.Series(dict(out), name=name).sort_index(), hist


# =============================================================================== MOMENTUM
def momentum_signal(R: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """Cumulative log return over months t-lookback .. t-skip-1 (12-1: t-12..t-2), known at start of t."""
    win = lookback - skip
    return np.log1p(R).rolling(win, min_periods=win).sum().shift(skip + 1)


def decile_portfolios(R: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sig, n = momentum_signal(R, cfg.mom_lookback, cfg.mom_skip), cfg.n_deciles
    cols = [f"P{i + 1}" for i in range(n)]
    out = pd.DataFrame(np.nan, index=R.index, columns=cols)
    for date, s in sig.iterrows():
        s = s.dropna()
        if len(s) < n:
            continue
        bucket = pd.qcut(s.rank(method="first"), n, labels=False)
        rets = R.loc[date, s.index]
        for i in range(n):
            m = rets[bucket == i]
            if m.notna().any():
                out.loc[date, cols[i]] = m.mean()
    return out.dropna(how="all")


def momentum_summary(P: pd.DataFrame, mkt: pd.Series, rf: pd.Series, min_obs: int):
    rf = rf.reindex(P.index)
    mkt_ex = (mkt - rf.reindex(mkt.index)).reindex(P.index)
    rows = {c: {"mean_monthly": P[c].mean(), "std_monthly": P[c].std(), "ann_mean": P[c].mean() * 12,
                "ann_std": P[c].std() * np.sqrt(12),
                "sharpe_ann": (P[c] - rf).mean() / P[c].std() * np.sqrt(12),
                **ols_capm(P[c] - rf, mkt_ex, min_obs)} for c in P.columns}
    summ = pd.DataFrame(rows).T
    wml = (P.iloc[:, -1] - P.iloc[:, 0]).rename("WML")          # zero-cost: no rf subtracted
    summ.loc["WML"] = {"mean_monthly": wml.mean(), "std_monthly": wml.std(), "ann_mean": wml.mean() * 12,
                       "ann_std": wml.std() * np.sqrt(12), "sharpe_ann": wml.mean() / wml.std() * np.sqrt(12),
                       **ols_capm(wml, mkt_ex, min_obs)}
    summ["alpha_ann_pct"] = summ["alpha"] * 12 * 100
    rho, p = sps.spearmanr(np.arange(1, len(P.columns) + 1), summ.loc[P.columns, "mean_monthly"].astype(float))
    t_mean = wml.mean() / (wml.std() / np.sqrt(wml.notna().sum()))
    return summ, wml, {"spearman_rho": float(rho), "spearman_p": float(p), "wml_mean_t_stat": float(t_mean)}


# =============================================================================== PLOTS
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_descriptive(stats, corr, mean_corr, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(stats["mean"] * 100, bins=15, color=BLUE, edgecolor="white")
    ax[0].set(title="Distribution of mean monthly returns", xlabel="Mean monthly return (%)",
              ylabel="Number of stocks")
    im = ax[1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_title(f"Correlation matrix (avg pairwise ρ = {mean_corr:.2f})")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    _save(fig, path)


def plot_sml(capm, R, mkt, rf_m, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(capm["beta"].astype(float), R.mean() * 12 * 100, s=32, alpha=0.75, color=BLUE,
               edgecolor="navy", label="Stocks")
    b = np.linspace(capm["beta"].min() - 0.1, capm["beta"].max() + 0.1, 50)
    ax.plot(b, (rf_m + b * (mkt.mean() - rf_m)) * 12 * 100, "--", color=RED, lw=2, label="SML (CAPM)")
    ax.set(xlabel="Market beta", ylabel="Mean annual return (%)", title="Security Market Line")
    ax.legend()
    _save(fig, path)


def plot_oos(series: dict, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"Tangency (unconstrained)": (BLUE, "-"), "Tangency (long-only)": (GREEN, "-"),
              "Equal-weighted (1/N)": (GREY, "--")}
    for name, s in series.items():
        c, ls = colors[name]
        ax.plot(s.index, (1 + s).cumprod(), label=name, color=c, ls=ls, lw=2)
    ax.set(ylabel="Growth of 1 (log scale)", yscale="log",
           title="Out-of-sample performance (rolling window, annual rebalancing)")
    ax.legend()
    _save(fig, path)


def plot_momentum(summ, n, path):
    d = [f"P{i + 1}" for i in range(n)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].bar(range(1, n + 1), summ.loc[d, "ann_mean"].astype(float) * 100, color=BLUE, edgecolor="white")
    ax[0].set(xlabel="Momentum decile (1 = losers, 10 = winners)", ylabel="Mean annual return (%)",
              title="Mean return by momentum decile", xticks=range(1, n + 1))
    al = summ.loc[d, "alpha_ann_pct"].astype(float)
    ax[1].bar(range(1, n + 1), al, color=[GREEN if a > 0 else RED for a in al], edgecolor="white")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set(xlabel="Momentum decile", ylabel="CAPM alpha (annualised %)",
              title="CAPM alpha by momentum decile", xticks=range(1, n + 1))
    _save(fig, path)


# =============================================================================== PIPELINE
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(ds: Dataset, cfg: Config, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    R = ds.returns
    rf = pd.Series(cfg.rf_monthly, index=R.index, name="rf")
    mkt = market_proxy(R, ds.mcap)
    S: dict = {"n_stocks": R.shape[1], "n_months": len(R), "start": str(R.index.min().date()),
               "end": str(R.index.max().date()), "rf_monthly": cfg.rf_monthly}

    log.info("1/4 Descriptive statistics")
    stats, corr = descriptive_stats(R, cfg.rf_monthly), R.corr()
    mc = mean_pairwise_corr(corr)
    stats.to_csv(out_dir / "q1_descriptive.csv")
    R.cov().to_csv(out_dir / "q1_covariance.csv")
    corr.to_csv(out_dir / "q1_correlation.csv")
    plot_descriptive(stats, corr, mc, fig_dir / "q1_descriptive.png")
    S["q1"] = {"avg_mean_monthly": R.mean().mean(), "avg_std_monthly": R.std().mean(), "avg_pairwise_corr": mc}

    log.info("2/4 Tangency portfolio, CAPM and GRS test")
    full = R.loc[:, R.notna().sum() >= cfg.min_obs]
    w_tan = tangency_weights(full.mean(), full.cov(), cfg.rf_monthly)
    S["q2_tangency_in_sample"] = portfolio_summary((full.fillna(0.0) * w_tan).sum(axis=1), cfg.rf_monthly)
    w_tan.sort_values(ascending=False).to_csv(out_dir / "q2_tangency_weights.csv", header=["weight"])
    capm = capm_table(R, rf, mkt, cfg.min_obs)
    capm.to_csv(out_dir / "q2_capm_per_stock.csv")
    S["q2_capm"] = {"mean_alpha_monthly": capm["alpha"].mean(), "mean_beta": capm["beta"].mean(),
                    "mean_r2": capm["r2"].mean(), "n_sig_alpha_5pct": int((capm["t_alpha"].abs() > 1.96).sum()),
                    "n_regressions": len(capm)}
    S["q2_grs"] = grs_test(R, rf, mkt)
    plot_sml(capm, R, mkt, cfg.rf_monthly, fig_dir / "q2_sml.png")

    log.info("3/4 Out-of-sample backtest (%d-year window)", cfg.window_years)
    oos_u, hist_u = rolling_oos(R, cfg, False)
    oos_l, hist = rolling_oos(R, cfg, True)
    ew = R.reindex(oos_u.index).mean(axis=1).rename("equal_weight")
    oos = pd.concat([oos_u, oos_l, ew], axis=1)
    oos.to_csv(out_dir / "q4_oos_returns.csv")
    S["q4_oos"] = {c: portfolio_summary(oos[c], cfg.rf_monthly) for c in oos.columns}
    S["q4_oos"]["n_rebalances"] = len(hist)
    S["q4_oos"]["max_gross_leverage_unconstrained"] = float(max(w.abs().sum() for w in hist_u.values()))
    S["q4_oos"]["unconstrained_fallbacks_to_1_over_N"] = int(
        sum(np.allclose(w, 1.0 / len(w)) for w in hist_u.values()))
    plot_oos({"Tangency (unconstrained)": oos_u, "Tangency (long-only)": oos_l,
              "Equal-weighted (1/N)": ew}, fig_dir / "q4_oos_cumulative.png")

    log.info("4/4 Momentum deciles")
    P = decile_portfolios(R, cfg)
    P.to_csv(out_dir / "q5_momentum_portfolio_returns.csv")
    mom, wml, tests = momentum_summary(P, mkt, rf, cfg.min_obs)
    mom.to_csv(out_dir / "q5_momentum_summary.csv")
    S["q5_momentum"] = {**tests, "wml_mean_monthly": wml.mean(), "wml_alpha_monthly": mom.loc["WML", "alpha"],
                        "wml_t_alpha": mom.loc["WML", "t_alpha"], "wml_beta": mom.loc["WML", "beta"]}
    plot_momentum(mom, cfg.n_deciles, fig_dir / "q5_momentum.png")

    (out_dir / "summary.json").write_text(json.dumps(_clean(S), indent=2))
    return S


# =============================================================================== SYNTHETIC DATA
def make_synthetic(path: Path, n_stocks: int = 60, n_months: int = 240, seed: int = 42) -> Path:
    """Synthetic workbook with the same layout as the real dataset (for demos and tests)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    tick = [f"STOCK_{i:02d}" for i in range(n_stocks)]
    mkt = rng.normal(0.007, 0.04, n_months)
    R = rng.normal(0, 0.002, n_stocks) + np.outer(mkt, rng.uniform(0.5, 1.5, n_stocks)) \
        + rng.normal(0, 0.06, (n_months, n_stocks))
    R[1:] += 0.05 * R[:-1]
    R = pd.DataFrame(R, index=dates, columns=tick)
    for k in rng.choice(n_stocks, 8, replace=False):          # late listings / early delistings
        R.iloc[:rng.integers(0, 60), k] = np.nan
        R.iloc[rng.integers(n_months - 40, n_months):, k] = np.nan
    mcap = pd.DataFrame(rng.lognormal(8, 1, n_stocks) * (1 + R.fillna(0)).cumprod().to_numpy(),
                        index=dates, columns=tick).where(R.notna())
    info = pd.DataFrame({"FULL NAME": tick, "COUNTRY": rng.choice(["US", "DE", "FR", "GB", "JP"], n_stocks),
                         "ECONOMIC SECTOR": rng.choice(["Technology", "Financials", "Energy", "Health Care"], n_stocks)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        info.to_excel(xw, sheet_name="STOCKS", index=False)
        mcap.to_excel(xw, sheet_name="market_cap")
        R.to_excel(xw, sheet_name=" returns")                  # leading space mirrors the real file
    return path


# =============================================================================== SELF-TESTS
def selftest() -> None:
    cfg = Config()
    rng = np.random.default_rng(0)
    t, n = 180, 40
    idx = pd.date_range("2005-01-31", periods=t, freq="ME")
    m = rng.normal(0.006, 0.04, t)
    R = pd.DataFrame(np.outer(m, rng.uniform(0.6, 1.4, n)) + rng.normal(0, 0.05, (t, n)), index=idx,
                     columns=[f"S{i}" for i in range(n)])
    mc = pd.DataFrame(rng.uniform(1, 10, (t, n)), index=idx, columns=R.columns)

    x = pd.Series(rng.normal(0, 0.04, 2000))
    r = ols_capm(0.002 + 1.3 * x + pd.Series(rng.normal(0, 0.01, 2000)), x)
    assert abs(r["alpha"] - 0.002) < 5e-4 and abs(r["beta"] - 1.3) < 0.02, "OLS"

    for lo in (False, True):
        w = tangency_weights(R.mean(), R.cov(), 0.0025, lo)
        assert abs(w.sum() - 1) < 1e-8 and (not lo or (w >= -1e-9).all()), "weights"

    mc2 = mc.copy(); mc2.iloc[-1] *= 100
    assert np.allclose(market_proxy(R, mc), market_proxy(R, mc2), equal_nan=True), "market look-ahead"

    sig = momentum_signal(R, 12, 1)
    assert np.allclose(sig.iloc[50], np.log1p(R.iloc[38:49]).sum()), "momentum window"   # t-12..t-2

    P1 = decile_portfolios(R, cfg)
    R2 = R.copy(); R2.iloc[-1] = 0.5
    assert np.allclose(P1.iloc[:-1], decile_portfolios(R2, cfg).iloc[:-1], equal_nan=True), "momentum look-ahead"

    oos, wh = rolling_oos(R, cfg, True)
    assert oos.index.min() > R.index[cfg.window_years * 12 - 1] and len(wh) >= 1, "oos"

    assert (rolling_oos(R, cfg, False)[0] >= -1.0).all(), "returns below -100%"
    tt, nn = 600, 10
    ix = pd.date_range("1970-01-31", periods=tt, freq="ME")
    mk = pd.Series(rng.normal(0.006, 0.04, tt), index=ix)
    Rg = pd.DataFrame(np.outer(mk, rng.uniform(0.7, 1.3, nn)) + rng.normal(0, 0.03, (tt, nn)), index=ix)
    assert grs_test(Rg, pd.Series(0.0, index=ix), mk)["p_value"] > 0.01, "GRS"
    print("All self-tests passed.")


# =============================================================================== REPO FILES
README_EN = """# Equity Portfolio Analysis

A reproducible, single-file Python toolkit for empirical asset-pricing and portfolio analysis on a
monthly panel of stocks: from descriptive statistics to mean-variance optimisation, CAPM/GRS testing,
out-of-sample backtesting and momentum sorts.

> Originally written for the *Intelligent Methods for Financial Data Analysis* course (2026) and
> refactored into a tested, documented tool. Author: @AUTHOR@

## What it does

| Step | Analysis | Method |
|------|----------|--------|
| 1 | Descriptive statistics | Mean, volatility, skewness, kurtosis, Sharpe, covariance and correlation |
| 2 | Tangency portfolio + CAPM | Closed-form max-Sharpe weights, per-stock CAPM regressions, joint **GRS test** (exact F p-value) |
| 3 | Out-of-sample backtest | Rolling 5-year window, annual rebalancing, drifting weights; unconstrained vs long-only tangency vs 1/N |
| 4 | Momentum (12-1) | 10 decile portfolios, CAPM alphas, winners-minus-losers (WML), Spearman monotonicity test |

## Quick start

```bash
git clone @REPO@
cd @FOLDER@
python equity_portfolio_analysis.py --demo        # synthetic data, no private file needed
python equity_portfolio_analysis.py --selftest    # built-in unit tests
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
"""

README_EL = """# Ανάλυση Χαρτοφυλακίου Μετοχών

Αυτόνομο εργαλείο Python (ένα αρχείο) για εμπειρική ανάλυση τιμολόγησης περιουσιακών στοιχείων σε μηνιαία
δεδομένα μετοχών: περιγραφική στατιστική, βελτιστοποίηση μέσου-διακύμανσης, CAPM/GRS, backtest
εκτός δείγματος και στρατηγικές momentum.

> Αρχικά φτιάχτηκε για το μάθημα *Έξυπνες Μέθοδοι Ανάλυσης Οικονομικών Δεδομένων* (2026).
> Συγγραφέας: @AUTHOR@

## Τι κάνει

1. **Περιγραφική στατιστική**: μέσος, τυπική απόκλιση, ασυμμετρία, κύρτωση, Sharpe, συσχετίσεις.
2. **Tangency χαρτοφυλάκιο + CAPM**: βάρη μέγιστου Sharpe, CAPM ανά μετοχή, κοινό τεστ **GRS**.
3. **Backtest εκτός δείγματος**: κυλιόμενο παράθυρο 5 ετών, ετήσιο rebalancing· σύγκριση tangency χωρίς
   περιορισμούς, long-only και 1/N.
4. **Momentum (12-1)**: 10 δεκατημόρια, CAPM alphas, winners-minus-losers (WML), τεστ Spearman.

## Γρήγορη εκκίνηση

```bash
python equity_portfolio_analysis.py --demo        # συνθετικά δεδομένα
python equity_portfolio_analysis.py --selftest    # ενσωματωμένοι έλεγχοι
python equity_portfolio_analysis.py --data data/to_arxeio_sou.xlsx
```

Τα πακέτα που λείπουν εγκαθίστανται αυτόματα. Τα αποτελέσματα (CSV, `summary.json`, γραφήματα)
αποθηκεύονται στον φάκελο `results/`.

## Σημαντικές παρατηρήσεις

- Χωρίς look-ahead: οι έλεγχοι το επιβεβαιώνουν.
- Τα κενά δεδομένα **δεν** αντικαθίστανται με 0 στις εκτιμήσεις.
- Το tangency χωρίς περιορισμούς είναι ασταθές με 50 μετοχές και 60 παρατηρήσεις (ακραία μόχλευση),
  γι' αυτό παρουσιάζονται και το long-only και το 1/N.
- Το risk-free είναι σταθερό 3% ετησίως (`--rf` για αλλαγή).

## Άδεια
MIT
"""

GITIGNORE = """__pycache__/
*.pyc
.venv/
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
# private / licensed data and generated outputs are never committed
data/*
!data/.gitkeep
results/
"""

REQUIREMENTS = "numpy>=1.24\npandas>=2.0\nscipy>=1.10\nmatplotlib>=3.7\nopenpyxl>=3.1\n"

LICENSE_MIT = """MIT License

Copyright (c) @YEAR@ @AUTHOR@

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def init_repo(folder: Path, author: str, repo_url: str) -> None:
    """Write README (EN + EL), LICENSE, .gitignore, requirements.txt and data/ placeholder."""
    import datetime
    folder.mkdir(parents=True, exist_ok=True)
    fill = lambda t: (t.replace("@AUTHOR@", author).replace("@REPO@", repo_url)
                       .replace("@FOLDER@", repo_url.rstrip("/").split("/")[-1].removesuffix(".git"))
                       .replace("@YEAR@", str(datetime.date.today().year)))
    files = {"README.md": README_EN, "README.el.md": README_EL, "LICENSE": LICENSE_MIT,
             ".gitignore": GITIGNORE, "requirements.txt": REQUIREMENTS}
    for name, text in files.items():
        (folder / name).write_text(fill(text), encoding="utf-8")
    (folder / "data").mkdir(exist_ok=True)
    (folder / "data" / ".gitkeep").touch()
    print(f"Repository files written to {folder.resolve()}:")
    for name in [*files, "data/.gitkeep"]:
        print(f"  - {name}")
    print("\nNext: put equity_portfolio_analysis.py in this folder, then\n"
          "  git init && git add . && git commit -m 'Initial commit' && git push")


# =============================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Equity portfolio analysis toolkit.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path, help="Excel file with sheets STOCKS, market_cap, returns")
    g.add_argument("--demo", action="store_true", help="run on generated synthetic data")
    g.add_argument("--selftest", action="store_true", help="run built-in sanity tests")
    g.add_argument("--init-repo", action="store_true",
                   help="write README (EN+EL), LICENSE, .gitignore, requirements.txt for GitHub")
    ap.add_argument("--author", default="Zisis", help="name used in README/LICENSE (with --init-repo)")
    ap.add_argument("--repo-url", default="https://github.com/YOUR-USERNAME/equity-portfolio-analysis.git",
                    help="GitHub URL used in the README (with --init-repo)")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output directory (default: results)")
    ap.add_argument("--rf", type=float, default=Config.rf_annual, help="annual risk-free rate (default 0.03)")
    ap.add_argument("--n-stocks", type=int, default=Config.n_stocks)
    ap.add_argument("--window-years", type=int, default=Config.window_years)
    ap.add_argument("--max-leverage", type=float, default=None,
                    help="optional gross exposure cap for the unconstrained tangency portfolio")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s | %(message)s")
    if a.selftest:
        selftest()
        return 0
    if a.init_repo:
        init_repo(Path("."), a.author, a.repo_url)
        return 0

    cfg = Config(rf_annual=a.rf, n_stocks=a.n_stocks, window_years=a.window_years,
                 max_gross_leverage=a.max_leverage)
    path = a.data
    if a.demo:
        path = make_synthetic(Path(tempfile.gettempdir()) / "eqport_synthetic.xlsx")
        log.info("Using SYNTHETIC demo data (not real market data).")
    S = run(load_dataset(path, cfg), cfg, a.out)

    log.info("Done. Outputs in %s/", a.out)
    log.info("Tangency in-sample Sharpe: %.2f | GRS p-value: %.3f | WML mean: %.2f%%/month",
             S["q2_tangency_in_sample"]["sharpe_ann"], S["q2_grs"]["p_value"], S["q5_momentum"]["wml_mean_monthly"] * 100)
    log.info("Unconstrained portfolio: max gross leverage %.1fx; %d of %d rebalances used the 1/N fallback",
             S["q4_oos"]["max_gross_leverage_unconstrained"],
             S["q4_oos"]["unconstrained_fallbacks_to_1_over_N"], S["q4_oos"]["n_rebalances"])
    for k, v in S["q4_oos"].items():
        if isinstance(v, dict):
            log.info("OOS %-18s Sharpe %.2f | ann.ret %.1f%% | ann.vol %.1f%% | max DD %.0f%%",
                     k, v["sharpe_ann"], v["ann_mean"] * 100, v["ann_std"] * 100, v["max_drawdown"] * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
