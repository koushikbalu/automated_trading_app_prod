"""Run 10yr and 20yr backtests with current config (time stop OFF)
and compare against top Indian indexes, MFs, and momentum funds.

Usage:  python _bt_10_20_compare.py
"""
from __future__ import annotations

import logging
import sys
import io
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("bt_compare")
logger.setLevel(logging.INFO)

from backtest import (
    Config,
    download_ohlcv,
    load_config_from_yaml,
    run_backtest,
    _yf_tickers,
)
from constants import BROAD_UNIVERSE
from utils import annualized_return, annualized_vol, max_drawdown, sharpe_ratio


# ─── Benchmark tickers to compare against ───
# Indexes, MFs (via NAV proxies on yfinance), Momentum ETFs/Indexes
COMPARISON_TICKERS = {
    # Major Indexes
    "^NSEI":          "Nifty 50",
    "^CNX200":        "Nifty 200",
    "^CNXSC":         "Nifty Smallcap 100",
    "^CRSLDX":        "Nifty Midcap 100",
    # Momentum / Factor Indexes (if available)
    "NIFTY200MOMENTM30.NS": "Nifty200 Momentum 30",
    # Top Mutual Fund NAVs (tracked as ETFs/tickers on yfinance)
    "0P0000XVNI.BO":  "Parag Parikh Flexi Cap",
    "0P0001BAO4.BO":  "Quant Small Cap",
    "0P0000XVMX.BO":  "Mirae Asset Large Cap",
    "0P00009VR1.BO":  "HDFC Flexi Cap",
    "0P0000XVAA.BO":  "Axis Bluechip",
    "0P0000XVAN.BO":  "SBI Small Cap",
    "0P0000XVQ8.BO":  "Kotak Emerging Equity",
    # Momentum-specific MFs
    "0P000600FQ.BO":  "Motilal Oswal M50 ETF",
    "0P0001I2TQ.BO":  "ICICI Pru Alpha Fund",
}


def fetch_benchmark_data(start: str, end: str | None = None) -> dict[str, pd.Series]:
    """Download Adj Close for all comparison tickers."""
    tickers = list(COMPARISON_TICKERS.keys())
    logger.info("Downloading %d comparison tickers...", len(tickers))

    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    except Exception as e:
        logger.error("Download failed: %s", e)
        return {}

    result = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            key = (t, "Adj Close")
            if key in raw.columns:
                series = raw[key].dropna()
                if len(series) > 252:
                    result[t] = series
                else:
                    logger.warning("Insufficient data for %s (%d rows)", COMPARISON_TICKERS.get(t, t), len(series))
            else:
                logger.warning("No data for %s", COMPARISON_TICKERS.get(t, t))
    return result


def compute_benchmark_metrics(series: pd.Series, rf: float = 0.04) -> dict:
    """Compute CAGR, MaxDD, Sharpe, Vol from a price series."""
    rets = series.pct_change().dropna()
    return {
        "cagr": annualized_return(series),
        "max_dd": max_drawdown(series),
        "vol": annualized_vol(rets),
        "sharpe": sharpe_ratio(rets, rf),
    }


def run_strategy_backtest(start_date: str) -> dict:
    """Run the full production backtest from the given start date."""
    config = load_config_from_yaml()
    config = replace(config, start_date=start_date)
    logger.info("Running strategy backtest from %s ...", start_date)
    result = run_backtest(config)
    summary = result["summary"]

    def get_metric(name):
        row = summary[summary["Metric"] == name]
        if row.empty:
            return np.nan
        val = row["Strategy"].iloc[0]
        return float(val) if not isinstance(val, str) else val

    daily = result["daily_results"]
    curve = daily.set_index("date")["strategy_value"] if "date" in daily.columns else daily["strategy_value"]

    return {
        "cagr": get_metric("CAGR"),
        "max_dd": get_metric("Max Drawdown"),
        "vol": get_metric("Annualized Volatility"),
        "sharpe": get_metric("Sharpe (rf=4%)"),
        "sortino": get_metric("Sortino (rf=4%)"),
        "calmar": get_metric("Calmar Ratio"),
        "final_value": get_metric("Final Value"),
        "win_rate": get_metric("Win Rate (by trade)"),
        "profit_factor": get_metric("Profit Factor"),
        "avg_exposure": get_metric("Average Exposure"),
        "longest_dd": get_metric("Longest Drawdown (days)"),
        "monthly_hit_rate": get_metric("Monthly Hit Rate vs Nifty200"),
    }


def print_comparison_table(strat_10: dict, strat_20: dict,
                           bench_10: dict[str, dict], bench_20: dict[str, dict]):
    """Print the grand comparison table."""

    def fmt_pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v*100:.2f}%"

    def fmt_shp(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.2f}"

    # ── 10-Year Table ──
    print("\n" + "=" * 120)
    print("10-YEAR COMPARISON (Apr 2016 - Apr 2026)")
    print("=" * 120)
    print(f"{'Name':<35} {'CAGR':>8} {'Max DD':>9} {'Volatility':>11} {'Sharpe':>8} {'Category':>20}")
    print("-" * 120)

    print(f"{'>> OUR STRATEGY <<':<35} {fmt_pct(strat_10['cagr']):>8} {fmt_pct(strat_10['max_dd']):>9} "
          f"{fmt_pct(strat_10['vol']):>11} {fmt_shp(strat_10['sharpe']):>8} {'Momentum (Active)':>20}")

    sorted_bench_10 = sorted(bench_10.items(), key=lambda x: x[1].get("cagr", 0), reverse=True)
    for ticker, metrics in sorted_bench_10:
        name = COMPARISON_TICKERS.get(ticker, ticker)[:34]
        cat = "Index" if ticker.startswith("^") else "Momentum" if "Moment" in name or "M50" in name or "Alpha" in name else "Mutual Fund"
        print(f"{name:<35} {fmt_pct(metrics['cagr']):>8} {fmt_pct(metrics['max_dd']):>9} "
              f"{fmt_pct(metrics['vol']):>11} {fmt_shp(metrics['sharpe']):>8} {cat:>20}")

    # ── 20-Year Table ──
    print("\n\n" + "=" * 120)
    print("20-YEAR COMPARISON (Apr 2006 - Apr 2026)")
    print("=" * 120)
    print(f"{'Name':<35} {'CAGR':>8} {'Max DD':>9} {'Volatility':>11} {'Sharpe':>8} {'Category':>20}")
    print("-" * 120)

    print(f"{'>> OUR STRATEGY <<':<35} {fmt_pct(strat_20['cagr']):>8} {fmt_pct(strat_20['max_dd']):>9} "
          f"{fmt_pct(strat_20['vol']):>11} {fmt_shp(strat_20['sharpe']):>8} {'Momentum (Active)':>20}")

    sorted_bench_20 = sorted(bench_20.items(), key=lambda x: x[1].get("cagr", 0), reverse=True)
    for ticker, metrics in sorted_bench_20:
        name = COMPARISON_TICKERS.get(ticker, ticker)[:34]
        cat = "Index" if ticker.startswith("^") else "Momentum" if "Moment" in name or "M50" in name or "Alpha" in name else "Mutual Fund"
        print(f"{name:<35} {fmt_pct(metrics['cagr']):>8} {fmt_pct(metrics['max_dd']):>9} "
              f"{fmt_pct(metrics['vol']):>11} {fmt_shp(metrics['sharpe']):>8} {cat:>20}")

    # ── Alpha Summary ──
    print("\n\n" + "=" * 80)
    print("ALPHA OVER BENCHMARKS")
    print("=" * 80)

    our_10 = strat_10["cagr"]
    our_20 = strat_20["cagr"]

    print(f"\n{'Benchmark':<35} {'10yr Alpha':>12} {'20yr Alpha':>12}")
    print("-" * 60)

    all_tickers = set(list(bench_10.keys()) + list(bench_20.keys()))
    for t in sorted(all_tickers, key=lambda x: COMPARISON_TICKERS.get(x, x)):
        name = COMPARISON_TICKERS.get(t, t)[:34]
        a10 = (our_10 - bench_10[t]["cagr"]) * 100 if t in bench_10 and our_10 is not None else np.nan
        a20 = (our_20 - bench_20[t]["cagr"]) * 100 if t in bench_20 and our_20 is not None else np.nan
        a10_s = f"{a10:+.2f}pp" if pd.notna(a10) else "N/A"
        a20_s = f"{a20:+.2f}pp" if pd.notna(a20) else "N/A"
        print(f"{name:<35} {a10_s:>12} {a20_s:>12}")

    # ── Strategy detail side-by-side ──
    print("\n\n" + "=" * 80)
    print("STRATEGY DETAIL: 10yr vs 20yr")
    print("=" * 80)

    metrics_list = [
        ("CAGR", "cagr", True),
        ("Max Drawdown", "max_dd", True),
        ("Volatility", "vol", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
        ("Win Rate", "win_rate", True),
        ("Profit Factor", "profit_factor", False),
        ("Avg Exposure", "avg_exposure", True),
        ("Longest DD (days)", "longest_dd", False),
        ("Monthly Hit Rate vs Nifty200", "monthly_hit_rate", True),
    ]

    print(f"{'Metric':<35} {'10-Year':>15} {'20-Year':>15}")
    print("-" * 65)
    for label, key, is_pct in metrics_list:
        v10 = strat_10.get(key, np.nan)
        v20 = strat_20.get(key, np.nan)
        if is_pct:
            s10 = f"{v10*100:.2f}%" if pd.notna(v10) else "N/A"
            s20 = f"{v20*100:.2f}%" if pd.notna(v20) else "N/A"
        else:
            s10 = f"{v10:.2f}" if pd.notna(v10) else "N/A"
            s20 = f"{v20:.2f}" if pd.notna(v20) else "N/A"
        print(f"{label:<35} {s10:>15} {s20:>15}")

    fv10 = strat_10.get("final_value", np.nan)
    fv20 = strat_20.get("final_value", np.nan)
    print(f"{'Final Value':<35} {'Rs '+str(int(fv10//1e5))+'L' if pd.notna(fv10) else 'N/A':>15} "
          f"{'Rs '+str(int(fv20//1e5))+'L' if pd.notna(fv20) else 'N/A':>15}")


def main():
    t0 = time.time()

    # ── Run strategy backtests ──
    logger.info("=== Running 10-year strategy backtest ===")
    strat_10 = run_strategy_backtest("2016-04-01")
    logger.info("10yr CAGR: %.2f%%", strat_10["cagr"] * 100)

    logger.info("=== Running 20-year strategy backtest ===")
    strat_20 = run_strategy_backtest("2006-04-01")
    logger.info("20yr CAGR: %.2f%%", strat_20["cagr"] * 100)

    # ── Fetch benchmark data ──
    logger.info("=== Fetching benchmark data (10yr) ===")
    bench_series_10 = fetch_benchmark_data("2016-04-01")
    bench_10 = {}
    for t, series in bench_series_10.items():
        try:
            bench_10[t] = compute_benchmark_metrics(series)
        except Exception as e:
            logger.warning("Failed to compute metrics for %s: %s", COMPARISON_TICKERS.get(t, t), e)

    logger.info("=== Fetching benchmark data (20yr) ===")
    bench_series_20 = fetch_benchmark_data("2006-04-01")
    bench_20 = {}
    for t, series in bench_series_20.items():
        try:
            bench_20[t] = compute_benchmark_metrics(series)
        except Exception as e:
            logger.warning("Failed to compute metrics for %s: %s", COMPARISON_TICKERS.get(t, t), e)

    # ── Print results ──
    print_comparison_table(strat_10, strat_20, bench_10, bench_20)

    total = time.time() - t0
    print(f"\n\nTotal runtime: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
