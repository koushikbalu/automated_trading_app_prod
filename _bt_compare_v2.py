"""Compare our strategy (10yr + 20yr) against Indian indexes, MFs, momentum funds.
Downloads each benchmark individually to avoid yfinance multi-download issues.

Usage:  python _bt_compare_v2.py
"""
from __future__ import annotations

import logging
import sys
import io
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("compare")
logger.setLevel(logging.INFO)

from utils import annualized_return, annualized_vol, max_drawdown, sharpe_ratio

BENCHMARKS = [
    # (ticker, display_name, category)
    ("^NSEI",               "Nifty 50",                  "Index"),
    ("^NSEBANK",            "Nifty Bank",                "Index"),
    ("^CNXSC",              "Nifty Smallcap 100",        "Index"),
    ("^CNXMID",             "Nifty Midcap 150",          "Index"),
    ("NIFTYBEES.NS",        "Nifty BeES ETF",            "Index ETF"),
    ("JUNIORBEES.NS",       "Junior BeES (NN50 ETF)",    "Index ETF"),
    ("MOTILALM50.NS",       "Motilal Oswal M50 ETF",    "Momentum ETF"),
    ("0P0000XVNI.BO",       "Parag Parikh Flexi Cap",    "Mutual Fund"),
    ("0P0001BAO4.BO",       "Quant Small Cap Fund",      "Mutual Fund"),
    ("0P0000XVMX.BO",       "Mirae Asset Large Cap",     "Mutual Fund"),
    ("0P0000XVAN.BO",       "SBI Small Cap Fund",        "Mutual Fund"),
    ("0P0000XVQ8.BO",       "Kotak Emerging Equity",     "Mutual Fund"),
    ("0P0000XVC2.BO",       "HDFC Mid-Cap Opp Fund",     "Mutual Fund"),
    ("0P0000XVAA.BO",       "Axis Bluechip Fund",        "Mutual Fund"),
    ("0P0000XVEZ.BO",       "Nippon India Small Cap",    "Mutual Fund"),
    ("0P0000XVO2.BO",       "ICICI Pru Bluechip Fund",   "Mutual Fund"),
]


def download_single(ticker: str, start: str, end: str = None) -> pd.Series | None:
    """Download a single ticker's Adj Close."""
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            if ("Adj Close", ticker) in data.columns:
                series = data[("Adj Close", ticker)].dropna()
            elif "Adj Close" in data.columns.get_level_values(0):
                series = data["Adj Close"].iloc[:, 0].dropna() if hasattr(data["Adj Close"], "iloc") else data["Adj Close"].dropna()
            else:
                return None
        else:
            if "Adj Close" not in data.columns:
                return None
            series = data["Adj Close"].dropna()
        if hasattr(series, "columns"):
            series = series.squeeze()
        if len(series) < 200:
            return None
        return series
    except Exception as e:
        logger.warning("  Download error for %s: %s", ticker, e)
        return None


def compute_metrics(series: pd.Series, rf: float = 0.04) -> dict:
    rets = series.pct_change().dropna()
    cagr = annualized_return(series)
    mdd = max_drawdown(series)
    vol = annualized_vol(rets)
    shp = sharpe_ratio(rets, rf)
    drf = (1 + rf) ** (1 / 252) - 1
    ds_std = rets[rets < 0].std(ddof=1)
    sortino = (rets - drf).mean() / ds_std * np.sqrt(252) if ds_std > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    years = (series.index[-1] - series.index[0]).days / 365.25
    final_mult = series.iloc[-1] / series.iloc[0]
    return {
        "cagr": cagr, "max_dd": mdd, "vol": vol, "sharpe": shp,
        "sortino": sortino, "calmar": calmar, "years": years, "mult": final_mult,
    }


def main():
    t0 = time.time()

    # Our strategy results (from the backtest output we just ran)
    # 10-year: 32.10% CAGR, -30.60% MaxDD, 20.29% vol, 1.31 Sharpe
    # 20-year: 29.75% CAGR, -31.08% MaxDD, 19.99% vol, 1.24 Sharpe
    our_strat = {
        "10yr": {"cagr": 0.3210, "max_dd": -0.3060, "vol": 0.2029, "sharpe": 1.31,
                 "sortino": 1.51, "calmar": 1.05, "mult": 2468_00_000 / 2_00_00_000},
        "20yr": {"cagr": 0.2975, "max_dd": -0.3108, "vol": 0.1999, "sharpe": 1.24,
                 "sortino": 1.42, "calmar": 0.96, "mult": 284_12_00_000 / 2_00_00_000},
    }

    # Download benchmarks for both periods
    bench_10: dict[str, dict] = {}
    bench_20: dict[str, dict] = {}

    for ticker, name, category in BENCHMARKS:
        logger.info("Downloading %s (%s) ...", name, ticker)

        # 10-year
        s10 = download_single(ticker, "2016-04-01")
        if s10 is not None:
            try:
                bench_10[ticker] = {**compute_metrics(s10), "name": name, "cat": category}
                logger.info("  10yr: CAGR=%.2f%%", bench_10[ticker]["cagr"] * 100)
            except Exception as e:
                logger.warning("  10yr metrics failed: %s", e)
        else:
            logger.warning("  10yr: No data")

        # 20-year
        s20 = download_single(ticker, "2006-04-01")
        if s20 is not None:
            try:
                bench_20[ticker] = {**compute_metrics(s20), "name": name, "cat": category}
                logger.info("  20yr: CAGR=%.2f%%", bench_20[ticker]["cagr"] * 100)
            except Exception as e:
                logger.warning("  20yr metrics failed: %s", e)
        else:
            logger.warning("  20yr: No data")

    # Also add Nifty 200 from our backtest benchmark
    # The backtest already computed benchmark CAGR. Let's download it separately.
    n200 = download_single("^CNX200", "2006-04-01")
    if n200 is not None:
        s10_n200 = n200[n200.index >= "2016-04-01"]
        if len(s10_n200) > 200:
            bench_10["^CNX200"] = {**compute_metrics(s10_n200), "name": "Nifty 200", "cat": "Index"}
        bench_20["^CNX200"] = {**compute_metrics(n200), "name": "Nifty 200", "cat": "Index"}

    # ── Print 10-Year Table ──
    print("\n" + "=" * 130)
    print("10-YEAR PERFORMANCE COMPARISON (Apr 2016 - Apr 2026)")
    print("Initial Capital: Rs 2 Cr")
    print("=" * 130)

    print(f"\n{'Rank':>4} {'Name':<32} {'Category':<18} {'CAGR':>8} {'MaxDD':>9} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Growth':>8}")
    print("-" * 130)

    # Combine our strategy + benchmarks, sort by CAGR
    all_10 = [("OUR STRATEGY", "Momentum (Active)", our_strat["10yr"])]
    for t, m in bench_10.items():
        all_10.append((m["name"], m["cat"], m))

    all_10.sort(key=lambda x: x[2].get("cagr", 0), reverse=True)

    for rank, (name, cat, m) in enumerate(all_10, 1):
        marker = ">>>" if name == "OUR STRATEGY" else "   "
        cagr_s = f"{m['cagr']*100:.2f}%"
        mdd_s = f"{m['max_dd']*100:.2f}%"
        vol_s = f"{m['vol']*100:.2f}%"
        shp_s = f"{m['sharpe']:.2f}"
        sort_s = f"{m.get('sortino', np.nan):.2f}" if pd.notna(m.get('sortino')) else "N/A"
        calm_s = f"{m.get('calmar', np.nan):.2f}" if pd.notna(m.get('calmar')) else "N/A"
        mult_s = f"{m.get('mult', np.nan):.1f}x" if pd.notna(m.get('mult')) else "N/A"
        print(f"{marker}{rank:>1} {name:<32} {cat:<18} {cagr_s:>8} {mdd_s:>9} {vol_s:>8} {shp_s:>8} {sort_s:>8} {calm_s:>8} {mult_s:>8}")

    # ── Print 20-Year Table ──
    print("\n\n" + "=" * 130)
    print("20-YEAR PERFORMANCE COMPARISON (Apr 2006 - Apr 2026)")
    print("Initial Capital: Rs 2 Cr")
    print("=" * 130)

    print(f"\n{'Rank':>4} {'Name':<32} {'Category':<18} {'CAGR':>8} {'MaxDD':>9} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Growth':>8}")
    print("-" * 130)

    all_20 = [("OUR STRATEGY", "Momentum (Active)", our_strat["20yr"])]
    for t, m in bench_20.items():
        all_20.append((m["name"], m["cat"], m))

    all_20.sort(key=lambda x: x[2].get("cagr", 0), reverse=True)

    for rank, (name, cat, m) in enumerate(all_20, 1):
        marker = ">>>" if name == "OUR STRATEGY" else "   "
        cagr_s = f"{m['cagr']*100:.2f}%"
        mdd_s = f"{m['max_dd']*100:.2f}%"
        vol_s = f"{m['vol']*100:.2f}%"
        shp_s = f"{m['sharpe']:.2f}"
        sort_s = f"{m.get('sortino', np.nan):.2f}" if pd.notna(m.get('sortino')) else "N/A"
        calm_s = f"{m.get('calmar', np.nan):.2f}" if pd.notna(m.get('calmar')) else "N/A"
        mult_s = f"{m.get('mult', np.nan):.1f}x" if pd.notna(m.get('mult')) else "N/A"
        print(f"{marker}{rank:>1} {name:<32} {cat:<18} {cagr_s:>8} {mdd_s:>9} {vol_s:>8} {shp_s:>8} {sort_s:>8} {calm_s:>8} {mult_s:>8}")

    # ── Alpha table ──
    print("\n\n" + "=" * 90)
    print("ALPHA OVER EACH BENCHMARK (Our Strategy minus Benchmark CAGR)")
    print("=" * 90)

    print(f"\n{'Benchmark':<32} {'Category':<18} {'10yr Alpha':>12} {'20yr Alpha':>12}")
    print("-" * 75)

    all_names = set()
    for t in set(list(bench_10.keys()) + list(bench_20.keys())):
        name = bench_10.get(t, bench_20.get(t, {})).get("name", t)
        cat = bench_10.get(t, bench_20.get(t, {})).get("cat", "")
        a10 = (our_strat["10yr"]["cagr"] - bench_10[t]["cagr"]) * 100 if t in bench_10 else np.nan
        a20 = (our_strat["20yr"]["cagr"] - bench_20[t]["cagr"]) * 100 if t in bench_20 else np.nan
        a10_s = f"{a10:+.2f}pp" if pd.notna(a10) else "N/A"
        a20_s = f"{a20:+.2f}pp" if pd.notna(a20) else "N/A"
        print(f"{name:<32} {cat:<18} {a10_s:>12} {a20_s:>12}")

    # ── Strategy summary ──
    print("\n\n" + "=" * 80)
    print("OUR STRATEGY SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<35} {'10-Year':>15} {'20-Year':>15}")
    print("-" * 65)
    print(f"{'CAGR':<35} {'32.10%':>15} {'29.75%':>15}")
    print(f"{'Max Drawdown':<35} {'-30.60%':>15} {'-31.08%':>15}")
    print(f"{'Volatility':<35} {'20.29%':>15} {'19.99%':>15}")
    print(f"{'Sharpe (rf=4%)':<35} {'1.31':>15} {'1.24':>15}")
    print(f"{'Sortino (rf=4%)':<35} {'1.51':>15} {'1.42':>15}")
    print(f"{'Calmar':<35} {'1.05':>15} {'0.96':>15}")
    print(f"{'Win Rate':<35} {'41.0%':>15} {'41.8%':>15}")
    print(f"{'Profit Factor':<35} {'1.43':>15} {'1.46':>15}")
    print(f"{'Rs 2Cr grows to':<35} {'Rs 24.68 Cr':>15} {'Rs 284.12 Cr':>15}")
    print(f"{'Growth Multiple':<35} {'12.3x':>15} {'142.1x':>15}")

    total = time.time() - t0
    print(f"\n\nTotal runtime: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
