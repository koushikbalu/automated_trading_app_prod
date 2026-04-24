"""20-year A/B test: 6 targeted improvements vs baseline.

Downloads data ONCE (2006-04-01), runs 7 backtests (1 baseline + 6 mods),
then runs combination tests of any winners.

Usage:  python _ab_20yr_6mods.py
"""
from __future__ import annotations

import logging
import sys
import io
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("ab20yr")
logger.setLevel(logging.INFO)

from backtest import (
    Config,
    download_ohlcv,
    load_config_from_yaml,
    _yf_tickers,
    _yf_sector_map,
)
from constants import BROAD_UNIVERSE, SECTOR_MAP
from utils import (
    annualized_return,
    annualized_vol,
    blended_weights,
    capped_inverse_vol_weights,
    compute_atr_df,
    max_drawdown,
    sharpe_ratio,
)


def run_bt(config: Config, raw_data: dict, *, mod_id: str = "baseline") -> dict:
    """Core backtest engine with mod hooks. Returns summary metrics dict."""

    sector_map = _yf_sector_map(SECTOR_MAP)
    close = raw_data["Adj Close"].copy()
    high = raw_data["High"].copy()
    low = raw_data["Low"].copy()
    volume = raw_data["Volume"].copy()

    benchmark = close[config.benchmark].dropna()
    cols_to_drop = [config.benchmark]
    if config.momentum_benchmark and config.momentum_benchmark in close.columns:
        cols_to_drop.append(config.momentum_benchmark)

    stocks_close_raw = close.drop(columns=cols_to_drop, errors="ignore").reindex(benchmark.index)
    stocks_close = stocks_close_raw.ffill(limit=config.ffill_limit)

    _missing_mask = stocks_close_raw.isna()
    _missing_streak = (
        _missing_mask
        .apply(lambda col: col.groupby((~col).cumsum()).cumsum())
        .fillna(0)
        .astype(int)
    )

    stocks_high = high.drop(columns=[config.benchmark], errors="ignore").reindex(benchmark.index)
    stocks_low = low.drop(columns=[config.benchmark], errors="ignore").reindex(benchmark.index)
    stocks_volume = volume.drop(columns=[config.benchmark], errors="ignore").reindex(benchmark.index).fillna(0)

    daily_returns = stocks_close.pct_change().fillna(0)
    benchmark_returns = benchmark.pct_change().fillna(0)

    traded_value = stocks_close * stocks_volume
    adv_126 = traded_value.rolling(126).mean()
    dma_20 = stocks_close.rolling(20).mean()
    dma_100 = stocks_close.rolling(100).mean()
    dma_200 = stocks_close.rolling(200).mean()
    vol_60 = daily_returns.rolling(60).std(ddof=1) * np.sqrt(252)

    atr = compute_atr_df(stocks_high, stocks_low, stocks_close, config.atr_window)

    bench_50 = benchmark.rolling(50).mean()
    bench_200 = benchmark.rolling(200).mean()
    bench_63ret = benchmark / benchmark.shift(63) - 1

    _month_groups = pd.Series(benchmark.index, index=benchmark.index).groupby(
        [benchmark.index.year, benchmark.index.month]
    )
    _offset = config.rebal_offset_from_end
    if _offset <= 1:
        _rebal_candidates = _month_groups.last()
    else:
        _rebal_candidates = _month_groups.apply(
            lambda g: g.iloc[-_offset] if len(g) >= _offset else g.iloc[-1]
        )
    rebalance_dates = set(_rebal_candidates.values)

    first_trading_day_of_month = pd.Series(
        benchmark.index, index=benchmark.index
    ).groupby([benchmark.index.year, benchmark.index.month]).first()
    first_td_set = set(first_trading_day_of_month.values)

    portfolio_value = config.initial_capital
    benchmark_value = config.initial_capital
    daily_cash_ret = (1 + config.annual_cash_yield) ** (1 / 252) - 1

    positions: dict[str, dict] = {}
    pending_exits: list[str] = []
    stopped_out_this_month: dict[str, float] = {}
    last_rebal_cand: pd.DataFrame = pd.DataFrame()
    last_rebal_month: tuple[int, int] = (0, 0)
    portfolio_peak = config.initial_capital
    circuit_breaker_active = False
    cb_risk_on_streak = 0

    daily_records: list[dict] = []
    trade_count = 0
    rt_open: dict[str, dict] = {}
    rt_returns: list[float] = []
    rt_hold_days: list[int] = []

    mw = config.momentum_weights
    start_index = max(252, 200)

    for i in range(start_index, len(benchmark.index)):
        current_date = benchmark.index[i]
        buy_turnover = 0.0
        sell_turnover = 0.0

        if (current_date in first_td_set and current_date.month == config.addition_month
                and config.annual_addition > 0):
            portfolio_value += config.annual_addition
            benchmark_value += config.annual_addition

        benchmark_value *= 1 + benchmark_returns.loc[current_date]

        # Regime
        above_200 = stocks_close.loc[current_date] > dma_200.loc[current_date]
        notna_count = above_200.notna().sum()
        breadth = above_200.sum() / notna_count if notna_count > 0 else np.nan

        bench_above_200 = (pd.notna(bench_200.loc[current_date])
                           and benchmark.loc[current_date] > bench_200.loc[current_date])
        full_risk_on = (bench_above_200
                        and (bench_50.loc[current_date] > bench_200.loc[current_date])
                        and (not config.require_positive_3m_return or bench_63ret.loc[current_date] > 0)
                        and (breadth > config.breadth_threshold))
        neutral = (not full_risk_on and bench_above_200
                   and (breadth > config.neutral_breadth_threshold))

        if full_risk_on:
            allocation_pct = 1.0
        elif neutral:
            allocation_pct = config.neutral_allocation_pct
        else:
            allocation_pct = 0.0

        portfolio_peak = max(portfolio_peak, portfolio_value)
        portfolio_dd = (portfolio_value / portfolio_peak) - 1 if portfolio_peak > 0 else 0.0

        # Circuit breaker
        if circuit_breaker_active:
            if full_risk_on:
                cb_risk_on_streak += 1
            else:
                cb_risk_on_streak = 0
            if cb_risk_on_streak >= config.cb_reset_days:
                circuit_breaker_active = False
                cb_risk_on_streak = 0
                portfolio_peak = portfolio_value
            else:
                allocation_pct = 0.0
        elif portfolio_dd < config.drawdown_circuit_breaker:
            circuit_breaker_active = True
            allocation_pct = 0.0
            for ticker in list(positions.keys()):
                sell_turnover += positions[ticker]["weight"]
                if ticker in rt_open:
                    p = rt_open.pop(ticker)
                    price = stocks_close.loc[current_date, ticker]
                    wavg = p["tc"] / p["tw"] if p["tw"] > 0 else price
                    rt_returns.append(price / wavg - 1 if wavg > 0 else 0)
                    rt_hold_days.append((current_date - p["ed"]).days)
                del positions[ticker]

        # Pending exits (T+1)
        for ticker in pending_exits:
            if ticker in positions:
                sell_turnover += positions[ticker]["weight"]
                if ticker in rt_open:
                    p = rt_open.pop(ticker)
                    price = stocks_close.loc[current_date, ticker]
                    wavg = p["tc"] / p["tw"] if p["tw"] > 0 else price
                    rt_returns.append(price * (1 - config.stop_exit_slippage) / wavg - 1)
                    rt_hold_days.append((current_date - p["ed"]).days)
                del positions[ticker]
        pending_exits = []

        # Force-exit stale data
        for ticker in list(positions.keys()):
            if _missing_streak.loc[current_date, ticker] > config.ffill_limit:
                sell_turnover += positions[ticker]["weight"]
                rt_open.pop(ticker, None)
                del positions[ticker]

        cur_ym = (current_date.year, current_date.month)
        if cur_ym != last_rebal_month and last_rebal_month != (0, 0):
            stopped_out_this_month = {}

        # Monthly rebalance
        if current_date in rebalance_dates:
            stopped_out_this_month = {}
            last_rebal_month = cur_ym

            liquid = adv_126.loc[current_date].dropna().sort_values(ascending=False)
            liquid = liquid[stocks_close.loc[current_date, liquid.index] >= config.min_price]
            liquid_universe = liquid.head(config.top_liquid_n).index.tolist()
            num_slots = int(round(config.top_momentum_n * allocation_pct))

            if num_slots > 0 and len(liquid_universe) > 0:
                lu = liquid_universe
                ret_1m = stocks_close.loc[current_date, lu] / stocks_close.shift(21).loc[current_date, lu] - 1
                ret_3m = stocks_close.loc[current_date, lu] / stocks_close.shift(63).loc[current_date, lu] - 1
                ret_6m = stocks_close.loc[current_date, lu] / stocks_close.shift(126).loc[current_date, lu] - 1
                ret_12m = stocks_close.shift(21).loc[current_date, lu] / stocks_close.shift(252).loc[current_date, lu] - 1

                raw_mom = mw[0] * ret_12m + mw[1] * ret_6m + mw[2] * ret_3m + mw[3] * ret_1m
                elig = ((stocks_close.loc[current_date, lu] > dma_100.loc[current_date, lu])
                        & (stocks_close.loc[current_date, lu] > dma_200.loc[current_date, lu]))

                vol_20 = stocks_volume.iloc[max(0, i - 20): i + 1].mean()
                vol_60_avg = stocks_volume.iloc[max(0, i - 60): i + 1].mean()
                rel_volume = (vol_20 / vol_60_avg.replace(0, np.nan)).reindex(lu)
                volume_ok = rel_volume >= rel_volume.median()

                cand = pd.DataFrame({
                    "score": raw_mom, "vol": vol_60.loc[current_date, lu],
                    "eligible": elig, "volume_ok": volume_ok,
                }).dropna()
                cand = cand[cand["eligible"] & cand["volume_ok"]]

                if config.max_vol_percentile < 1.0 and not cand.empty:
                    vol_cap = cand["vol"].quantile(config.max_vol_percentile)
                    cand = cand[cand["vol"] <= vol_cap]
                cand["risk_adj_score"] = cand["score"] / cand["vol"].replace(0, np.nan)
                cand = cand.dropna()

                if config.sector_downtrend_penalty < 1.0 and not cand.empty:
                    cand["_sector"] = cand.index.map(lambda t: sector_map.get(t, "Other"))
                    sector_median = cand.groupby("_sector")["score"].median()
                    bad = set(sector_median[sector_median < 0].index)
                    if bad:
                        mask = cand["_sector"].isin(bad)
                        cand.loc[mask, "risk_adj_score"] *= config.sector_downtrend_penalty
                    cand = cand.drop(columns=["_sector"])

                if config.continuity_bonus > 0:
                    for t in positions:
                        if t in cand.index:
                            cand.loc[t, "risk_adj_score"] += config.continuity_bonus

                cand = cand.sort_values("risk_adj_score", ascending=False)
                last_rebal_cand = cand.copy()

                sector_counts: dict[str, int] = {}
                max_per_sector = max(1, int(num_slots * config.max_sector_weight) + 1)
                filtered_idx: list[str] = []
                for t in cand.index:
                    sec = sector_map.get(t, "Other")
                    if sector_counts.get(sec, 0) < max_per_sector:
                        filtered_idx.append(t)
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1
                    if len(filtered_idx) >= num_slots:
                        break

                selected = filtered_idx
                if selected:
                    new_weights = blended_weights(
                        cand.loc[selected, "vol"],
                        cand.loc[selected, "risk_adj_score"],
                        config.max_weight, config.score_blend,
                    )
                    keep = new_weights[new_weights >= config.min_weight]
                    if not keep.empty:
                        new_weights = keep / keep.sum()
                        selected = list(keep.index)
                    else:
                        new_weights = pd.Series(dtype=float)
                        selected = []
                else:
                    new_weights = pd.Series(dtype=float)
            else:
                liquid_universe = []; selected = []; new_weights = pd.Series(dtype=float)

            existing = set(positions.keys())
            target = set(selected)

            for ticker in list(existing - target):
                sell_turnover += positions.get(ticker, {}).get("weight", 0)
                if ticker in rt_open:
                    p = rt_open.pop(ticker)
                    price = stocks_close.loc[current_date, ticker]
                    wavg = p["tc"] / p["tw"] if p["tw"] > 0 else price
                    rt_returns.append(price / wavg - 1 if wavg > 0 else 0)
                    rt_hold_days.append((current_date - p["ed"]).days)
                if ticker in positions:
                    del positions[ticker]
                trade_count += 1

            for ticker in selected:
                price = stocks_close.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]
                new_w = float(new_weights.get(ticker, 0))
                if ticker not in positions:
                    stop_price = None
                    if config.use_atr_trailing_stop and pd.notna(atr_val):
                        stop_price = float(price - config.atr_multiple * atr_val)
                    positions[ticker] = {
                        "weight": new_w, "entry_price": float(price),
                        "entry_date": current_date, "high_watermark": float(price),
                        "stop_price": stop_price, "losing_days": 0, "pyramid_count": 0,
                    }
                    buy_turnover += new_w
                    trade_count += 1
                    rt_open[ticker] = {"tc": float(price) * new_w, "tw": new_w, "ed": current_date}
                else:
                    old_w = positions[ticker]["weight"]
                    delta = new_w - old_w
                    positions[ticker]["weight"] = new_w
                    if delta > 0:
                        buy_turnover += delta
                        if ticker in rt_open:
                            rt_open[ticker]["tc"] += float(price) * delta
                            rt_open[ticker]["tw"] += delta
                    else:
                        sell_turnover += abs(delta)

        # Re-entry
        if config.re_entry_enabled and stopped_out_this_month and not last_rebal_cand.empty:
            top_ranked = set(last_rebal_cand.head(config.top_momentum_n).index)
            for ticker in list(stopped_out_this_month.keys()):
                if ticker in positions or ticker in pending_exits:
                    continue
                price = stocks_close.loc[current_date, ticker]
                ma20 = dma_20.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]
                if pd.isna(price) or pd.isna(ma20):
                    continue
                if price > ma20 and ticker in top_ranked:
                    orig_w = stopped_out_this_month.pop(ticker)
                    stop_price = None
                    if config.use_atr_trailing_stop and pd.notna(atr_val):
                        stop_price = float(price - config.atr_multiple * atr_val)
                    positions[ticker] = {
                        "weight": orig_w, "entry_price": float(price),
                        "entry_date": current_date, "high_watermark": float(price),
                        "stop_price": stop_price, "losing_days": 0, "pyramid_count": 0,
                    }
                    buy_turnover += orig_w
                    rt_open[ticker] = {"tc": float(price) * orig_w, "tw": orig_w, "ed": current_date}

        # Pyramiding
        if config.pyramid_enabled and not last_rebal_cand.empty and allocation_pct > 0 and current_date not in rebalance_dates:
            top_ranked = set(last_rebal_cand.head(config.top_momentum_n).index)
            for ticker in list(positions.keys()):
                if ticker in pending_exits:
                    continue
                pos = positions[ticker]
                if pos["pyramid_count"] >= config.pyramid_max:
                    continue
                if pos["weight"] + config.pyramid_add_pct > config.max_weight:
                    continue
                price = stocks_close.loc[current_date, ticker]
                if pd.isna(price):
                    continue
                gain = (price / pos["entry_price"]) - 1
                if gain >= config.pyramid_threshold_pct and ticker in top_ranked:
                    pos["pyramid_count"] += 1
                    pos["weight"] += config.pyramid_add_pct
                    buy_turnover += config.pyramid_add_pct
                    if config.pyramid_ratchet_stop:
                        be_stop = pos["entry_price"]
                        if pos["stop_price"] is None or pos["stop_price"] < be_stop:
                            pos["stop_price"] = be_stop
                    if ticker in rt_open:
                        rt_open[ticker]["tc"] += float(price) * config.pyramid_add_pct
                        rt_open[ticker]["tw"] += config.pyramid_add_pct

        # Min exposure floor
        if not last_rebal_cand.empty and allocation_pct > 0:
            pos_exposure = sum(p["weight"] for p in positions.values())
            if pos_exposure < config.min_exposure:
                needed = config.min_exposure - pos_exposure
                already_held = set(positions.keys()) | set(pending_exits)
                floor_cands = [
                    t for t in last_rebal_cand.index
                    if t not in already_held
                    and pd.notna(stocks_close.loc[current_date, t])
                    and stocks_close.loc[current_date, t] >= config.min_price
                ][:config.min_exposure_slots]
                if floor_cands:
                    fv = last_rebal_cand.loc[floor_cands, "vol"]
                    fw = capped_inverse_vol_weights(fv, config.max_weight)
                    fw = fw / fw.sum() * needed
                    for ticker in floor_cands:
                        fwt = float(fw.loc[ticker])
                        price = stocks_close.loc[current_date, ticker]
                        atr_val = atr.loc[current_date, ticker]
                        sp = None
                        if config.use_atr_trailing_stop and pd.notna(atr_val):
                            sp = float(price - config.atr_multiple * atr_val)
                        positions[ticker] = {
                            "weight": fwt, "entry_price": float(price),
                            "entry_date": current_date, "high_watermark": float(price),
                            "stop_price": sp, "losing_days": 0, "pyramid_count": 0,
                        }
                        buy_turnover += fwt
                        rt_open[ticker] = {"tc": float(price) * fwt, "tw": fwt, "ed": current_date}

        # Daily PnL + exit signals
        gross_ret = 0.0
        if len(positions) == 0:
            gross_ret = daily_cash_ret
        else:
            for ticker in list(positions.keys()):
                price = stocks_close.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]
                if pd.isna(price):
                    continue
                positions[ticker]["high_watermark"] = max(positions[ticker]["high_watermark"], float(price))

                if config.use_atr_trailing_stop and pd.notna(atr_val):
                    cs = positions[ticker]["high_watermark"] - config.atr_multiple * atr_val
                    if positions[ticker]["stop_price"] is None:
                        positions[ticker]["stop_price"] = float(cs)
                    else:
                        positions[ticker]["stop_price"] = max(positions[ticker]["stop_price"], float(cs))

                exit_reason = None
                if (config.use_atr_trailing_stop and positions[ticker]["stop_price"] is not None
                        and price < positions[ticker]["stop_price"]):
                    exit_reason = "ATR"
                if exit_reason is None and config.max_loss_pct is not None:
                    if (price / positions[ticker]["entry_price"]) - 1 < config.max_loss_pct:
                        exit_reason = "Hard"
                if price < positions[ticker]["entry_price"]:
                    positions[ticker]["losing_days"] += 1
                else:
                    positions[ticker]["losing_days"] = 0
                if config.max_hold_losing_days > 0 and exit_reason is None and positions[ticker]["losing_days"] >= config.max_hold_losing_days:
                    exit_reason = "Time"

                if exit_reason is not None:
                    stopped_out_this_month[ticker] = positions[ticker]["weight"]
                    pending_exits.append(ticker)
                    trade_count += 1

            pre_total = 0.0
            for ticker in positions:
                if "current_value" not in positions[ticker]:
                    positions[ticker]["current_value"] = portfolio_value * positions[ticker]["weight"]
                pre_total += positions[ticker]["current_value"]
            inv_frac = pre_total / portfolio_value if portfolio_value > 0 else 0.0

            for ticker in positions:
                dr = daily_returns.loc[current_date, ticker]
                wf = positions[ticker]["current_value"] / portfolio_value if portfolio_value > 0 else 0.0
                gross_ret += wf * dr
                positions[ticker]["current_value"] *= 1 + dr
            if inv_frac < 1:
                gross_ret += (1 - inv_frac) * daily_cash_ret

            post_total = sum(positions[t]["current_value"] for t in positions)
            if post_total > 0:
                for ticker in positions:
                    positions[ticker]["weight"] = positions[ticker]["current_value"] / post_total

        cost = buy_turnover * config.buy_side_cost + sell_turnover * config.sell_side_cost
        net_ret = gross_ret - cost
        portfolio_value *= 1 + net_ret

        if positions:
            tw = sum(positions[t]["weight"] for t in positions)
            for t in positions:
                positions[t]["current_value"] = portfolio_value * (positions[t]["weight"] / tw if tw > 0 else 0)

        pw = sum(positions[t]["weight"] for t in positions) if positions else 0.0
        daily_records.append({
            "date": current_date,
            "strat_ret": net_ret,
            "bench_ret": benchmark_returns.loc[current_date],
            "strat_val": portfolio_value,
            "bench_val": benchmark_value,
            "turnover": buy_turnover + sell_turnover,
            "n_pos": len(positions),
            "cash_w": max(0.0, 1 - pw) if positions else 1.0,
        })

    # Metrics
    df = pd.DataFrame(daily_records)
    df["date"] = pd.to_datetime(df["date"])
    curve = df.set_index("date")["strat_val"]
    bench = df.set_index("date")["bench_val"]
    sret = df.set_index("date")["strat_ret"]
    bret = df.set_index("date")["bench_ret"]

    cagr = annualized_return(curve)
    mdd = max_drawdown(curve)
    vol = annualized_vol(sret)
    shp = sharpe_ratio(sret, config.annual_cash_yield)
    drf = (1 + config.annual_cash_yield) ** (1 / 252) - 1
    ds = sret[sret < 0].std(ddof=1) * np.sqrt(252)
    sortino = (sret - drf).mean() / sret[sret < 0].std(ddof=1) * np.sqrt(252) if ds > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    pk = curve.cummax()
    in_dd = curve < pk
    dd_g = (~in_dd).cumsum()
    dd_d = in_dd.groupby(dd_g).sum()
    longest_dd = int(dd_d.max()) if len(dd_d) > 0 else 0

    # Yearly
    yearly_s = curve.resample("YE").last().pct_change().dropna()
    yearly_b = bench.resample("YE").last().pct_change().dropna()
    yearly_data = []
    for dt in yearly_s.index:
        sr = yearly_s.loc[dt]
        br = yearly_b.loc[dt] if dt in yearly_b.index else np.nan
        yearly_data.append({"year": dt.year, "strat": sr, "bench": br})

    # Rolling 3yr
    cv = curve.values
    ci = curve.index
    r3 = []
    for j in range(756, len(cv)):
        r3.append((cv[j] / cv[j - 756]) ** (1 / 3) - 1)

    # Roundtrip stats
    rr = pd.Series(rt_returns) if rt_returns else pd.Series(dtype=float)
    win_rate = (rr > 0).mean() if len(rr) > 0 else np.nan
    avg_winner = rr[rr > 0].mean() if (rr > 0).any() else 0.0
    avg_loser = rr[rr <= 0].mean() if (rr <= 0).any() else 0.0
    gp = rr[rr > 0].sum()
    gl = abs(rr[rr <= 0].sum())
    pf = gp / gl if gl > 0 else np.inf
    avg_hold = np.mean(rt_hold_days) if rt_hold_days else np.nan

    return {
        "cagr": cagr, "bench_cagr": annualized_return(bench),
        "max_dd": mdd, "bench_mdd": max_drawdown(bench),
        "longest_dd": longest_dd,
        "vol": vol, "sharpe": shp, "sortino": sortino, "calmar": calmar,
        "win_rate": win_rate, "avg_winner": avg_winner, "avg_loser": avg_loser,
        "profit_factor": pf, "avg_hold": avg_hold,
        "total_trades": trade_count, "total_roundtrips": len(rr),
        "avg_turnover": df["turnover"].mean(),
        "avg_positions": df["n_pos"].mean(),
        "avg_exposure": (1 - df["cash_w"]).mean(),
        "final_value": curve.iloc[-1],
        "yearly_data": yearly_data,
        "rolling_3yr": r3,
        "best_year": max(y["strat"] for y in yearly_data) if yearly_data else np.nan,
        "worst_year": min(y["strat"] for y in yearly_data) if yearly_data else np.nan,
        "pos_years": sum(1 for y in yearly_data if y["strat"] > 0),
        "total_years": len(yearly_data),
    }


# ─── Test definitions ───
# Each: (label, config_overrides_dict)
SOLO_TESTS = {
    "00_baseline":       ("Baseline (current config)", {}),
    "01_atr_2.5":        ("Wider ATR Stop (2.5x)", {"atr_multiple": 2.5}),
    "02_time_stop_10d":  ("Relaxed Time Stop (10d)", {"max_hold_losing_days": 10}),
    "03_neutral_85":     ("Higher Neutral Alloc (85%)", {"neutral_allocation_pct": 0.85}),
    "04_min_exp_20":     ("Lower Min Exposure (20%)", {"min_exposure": 0.20}),
    "05_top_n_12":       ("12 Stocks (from 10)", {"top_momentum_n": 12}),
    "06_time_stop_off":  ("Time Stop Disabled", {"max_hold_losing_days": 0}),
}


def print_comparison(results: dict, base_key: str = "00_baseline", tests: dict = None):
    if tests is None:
        tests = SOLO_TESTS
    base = results.get(base_key)
    if not base:
        print("ERROR: Baseline missing.")
        return

    print(f"\n{'#':<5} {'Modification':<35} {'CAGR':>7} {'dCAGR':>7} {'MaxDD':>8} {'dDD':>7} {'Sharpe':>7} "
          f"{'dShp':>6} {'Sort':>6} {'Calm':>6} {'WinR':>6} {'PF':>6} {'AvgHld':>7} {'LDD':>5} {'Verdict':>8}")
    print("-" * 145)

    for key in tests:
        r = results.get(key)
        if not r:
            continue
        lab = tests[key][0] if isinstance(tests[key], tuple) else tests[key]
        dc = (r["cagr"] - base["cagr"]) * 100
        dd = (r["max_dd"] - base["max_dd"]) * 100
        ds = r["sharpe"] - base["sharpe"]

        v = "WIN" if dc > 0.5 and ds >= 0 else "NEUTRAL" if abs(dc) <= 0.5 else "LOSE"
        if key == base_key:
            v = "--"

        print(
            f"{key[:5]:<5} {lab:<35} "
            f"{r['cagr']*100:>6.2f}% {dc:>+6.2f} "
            f"{r['max_dd']*100:>7.2f}% {dd:>+6.2f} "
            f"{r['sharpe']:>6.2f} {ds:>+5.2f} "
            f"{r['sortino']:>5.2f} {r['calmar']:>5.2f} "
            f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
            f"{r['avg_hold']:>6.1f}d {r['longest_dd']:>5d} "
            f"{v:>8}"
        )
    print("-" * 145)


def print_yearly(results: dict, tests: dict = None):
    if tests is None:
        tests = SOLO_TESTS
    keys = list(tests.keys())
    all_years = sorted({y["year"] for r in results.values() for y in r.get("yearly_data", [])})

    short = {k: (tests[k][0][:20] if isinstance(tests[k], tuple) else tests[k][:20]) for k in keys}

    header = f"{'Year':>6}"
    for k in keys:
        header += f" {short[k]:>20}"
    print(header)
    print("-" * (6 + 21 * len(keys)))

    for yr in all_years:
        row = f"{yr:>6}"
        for k in keys:
            r = results.get(k)
            if not r:
                row += f" {'N/A':>20}"
                continue
            yd = {y["year"]: y["strat"] for y in r["yearly_data"]}
            v = yd.get(yr, np.nan)
            row += f" {v*100:>19.2f}%" if pd.notna(v) else f" {'N/A':>20}"
        print(row)


def main():
    t0 = time.time()
    config = load_config_from_yaml()
    config = replace(config, start_date="2006-04-01")
    logger.info("20-year A/B test starting. Period: %s to present", config.start_date)

    # Download once
    yf_universe = _yf_tickers(BROAD_UNIVERSE)
    extra = [config.benchmark]
    if config.momentum_benchmark:
        extra.append(config.momentum_benchmark)
    tickers = yf_universe + extra

    logger.info("Downloading data for %d tickers...", len(tickers))
    raw_data = download_ohlcv(tickers, config.start_date, config.end_date)
    logger.info("Downloaded in %.0fs.", time.time() - t0)

    # ── Round 1: Solo tests ──
    results: dict[str, dict] = {}
    for key, (label, overrides) in SOLO_TESTS.items():
        t1 = time.time()
        cfg = replace(config, **overrides) if overrides else config
        logger.info("Running [%s] %s ...", key, label)
        try:
            r = run_bt(cfg, raw_data, mod_id=key)
            results[key] = r
            logger.info("  => CAGR=%.2f%% MaxDD=%.2f%% Sharpe=%.2f (%.0fs)",
                        r["cagr"]*100, r["max_dd"]*100, r["sharpe"], time.time()-t1)
        except Exception as e:
            logger.error("  => FAILED: %s", e)
            import traceback; traceback.print_exc()

    print("\n" + "=" * 145)
    print("ROUND 1: SOLO A/B TESTS (20 years, 2006-2026)")
    print("=" * 145)
    print_comparison(results)

    # Determine winners
    base = results["00_baseline"]
    winners = []
    for key in SOLO_TESTS:
        if key == "00_baseline":
            continue
        r = results.get(key)
        if r and (r["cagr"] - base["cagr"]) * 100 > 0.3 and r["sharpe"] >= base["sharpe"]:
            winners.append(key)

    print(f"\nWinners from Round 1: {winners if winners else 'None'}")

    # ── Round 2: Combination tests ──
    if len(winners) >= 2:
        combo_tests: dict[str, tuple[str, dict]] = {"00_baseline": ("Baseline", {})}
        # Each winner solo
        for w in winners:
            combo_tests[w] = SOLO_TESTS[w]

        # All pairs
        from itertools import combinations
        for a, b in combinations(winners, 2):
            combo_key = f"combo_{a[3:]}+{b[3:]}"
            la = SOLO_TESTS[a][0][:15]
            lb = SOLO_TESTS[b][0][:15]
            merged = {**SOLO_TESTS[a][1], **SOLO_TESTS[b][1]}
            combo_tests[combo_key] = (f"{la} + {lb}", merged)

        # All winners combined
        if len(winners) >= 3:
            all_merged = {}
            for w in winners:
                all_merged.update(SOLO_TESTS[w][1])
            combo_tests["combo_all_winners"] = ("All Winners Combined", all_merged)

        print("\n\n" + "=" * 145)
        print(f"ROUND 2: COMBINATION TESTS ({len(combo_tests) - 1} combos)")
        print("=" * 145)

        combo_results: dict[str, dict] = {"00_baseline": base}
        for key, (label, overrides) in combo_tests.items():
            if key == "00_baseline":
                continue
            if key in results:
                combo_results[key] = results[key]
                continue
            t1 = time.time()
            cfg = replace(config, **overrides) if overrides else config
            logger.info("Running [%s] %s ...", key, label)
            try:
                r = run_bt(cfg, raw_data, mod_id=key)
                combo_results[key] = r
                results[key] = r
                logger.info("  => CAGR=%.2f%% MaxDD=%.2f%% Sharpe=%.2f (%.0fs)",
                            r["cagr"]*100, r["max_dd"]*100, r["sharpe"], time.time()-t1)
            except Exception as e:
                logger.error("  => FAILED: %s", e)

        print_comparison(combo_results, tests=combo_tests)
    elif len(winners) == 1:
        print("\nOnly 1 winner -- no combination tests needed.")
    else:
        # Even if no clear "winners", test promising combos
        print("\nNo clear winners. Testing promising combinations anyway...")

        near_neutral = []
        for key in SOLO_TESTS:
            if key == "00_baseline":
                continue
            r = results.get(key)
            if r and abs((r["cagr"] - base["cagr"]) * 100) <= 2.0:
                near_neutral.append(key)

        combo_tests: dict[str, tuple[str, dict]] = {"00_baseline": ("Baseline", {})}
        if len(near_neutral) >= 2:
            from itertools import combinations
            for a, b in combinations(near_neutral[:4], 2):
                ck = f"combo_{a[3:]}+{b[3:]}"
                la = SOLO_TESTS[a][0][:15]
                lb = SOLO_TESTS[b][0][:15]
                merged = {**SOLO_TESTS[a][1], **SOLO_TESTS[b][1]}
                combo_tests[ck] = (f"{la} + {lb}", merged)

            if len(near_neutral) >= 3:
                all_m = {}
                for w in near_neutral[:4]:
                    all_m.update(SOLO_TESTS[w][1])
                combo_tests["combo_top_neutrals"] = ("Top Neutrals Combined", all_m)

        # Always test some hand-picked combos
        combo_tests["combo_atr25_time10"] = ("ATR 2.5x + TimeStop 10d", {"atr_multiple": 2.5, "max_hold_losing_days": 10})
        combo_tests["combo_atr25_neutral85"] = ("ATR 2.5x + Neutral 85%", {"atr_multiple": 2.5, "neutral_allocation_pct": 0.85})
        combo_tests["combo_atr25_n12"] = ("ATR 2.5x + 12 stocks", {"atr_multiple": 2.5, "top_momentum_n": 12})
        combo_tests["combo_neutral85_n12"] = ("Neutral 85% + 12 stocks", {"neutral_allocation_pct": 0.85, "top_momentum_n": 12})
        combo_tests["combo_atr25_time10_n85"] = ("ATR2.5+Time10+Neut85", {"atr_multiple": 2.5, "max_hold_losing_days": 10, "neutral_allocation_pct": 0.85})
        combo_tests["combo_atr25_notime_n85"] = ("ATR2.5+NoTime+Neut85", {"atr_multiple": 2.5, "max_hold_losing_days": 0, "neutral_allocation_pct": 0.85})
        combo_tests["combo_kitchen_sink"] = ("Kitchen Sink (all 6)", {
            "atr_multiple": 2.5, "max_hold_losing_days": 10,
            "neutral_allocation_pct": 0.85, "min_exposure": 0.20,
            "top_momentum_n": 12,
        })
        combo_tests["combo_kitchen_notime"] = ("KitchenSink+NoTimeStop", {
            "atr_multiple": 2.5, "max_hold_losing_days": 0,
            "neutral_allocation_pct": 0.85, "min_exposure": 0.20,
            "top_momentum_n": 12,
        })

        print(f"\n\n{'='*145}")
        print(f"ROUND 2: COMBINATION TESTS ({len(combo_tests)-1} combos)")
        print("=" * 145)

        combo_results: dict[str, dict] = {"00_baseline": base}
        for key, (label, overrides) in combo_tests.items():
            if key == "00_baseline":
                continue
            if key in results:
                combo_results[key] = results[key]
                continue
            t1 = time.time()
            cfg = replace(config, **overrides)
            logger.info("Running [%s] %s ...", key, label)
            try:
                r = run_bt(cfg, raw_data, mod_id=key)
                combo_results[key] = r
                results[key] = r
                logger.info("  => CAGR=%.2f%% MaxDD=%.2f%% Sharpe=%.2f (%.0fs)",
                            r["cagr"]*100, r["max_dd"]*100, r["sharpe"], time.time()-t1)
            except Exception as e:
                logger.error("  => FAILED: %s", e)

        print_comparison(combo_results, tests=combo_tests)

    # ── Year-by-year for top results ──
    # Collect top 5 by CAGR
    ranked = sorted(results.items(), key=lambda x: x[1]["cagr"], reverse=True)
    top5_keys = [k for k, _ in ranked[:5]]
    if "00_baseline" not in top5_keys:
        top5_keys.append("00_baseline")

    top5_tests = {}
    for k in top5_keys:
        if k in SOLO_TESTS:
            top5_tests[k] = SOLO_TESTS[k]
        elif k in combo_tests:
            top5_tests[k] = combo_tests[k]
        else:
            top5_tests[k] = (k, {})

    top5_results = {k: results[k] for k in top5_keys if k in results}

    print(f"\n\n{'='*145}")
    print("YEAR-BY-YEAR RETURNS (Top 5 configs + Baseline)")
    print("=" * 145)
    print_yearly(top5_results, tests=top5_tests)

    # ── Detailed stats for top 3 ──
    print(f"\n\n{'='*100}")
    print("DETAILED COMPARISON: TOP 3 vs BASELINE")
    print("=" * 100)

    top3_keys = [k for k, _ in ranked[:3]]
    if "00_baseline" not in top3_keys:
        top3_keys.append("00_baseline")

    for k in top3_keys:
        r = results[k]
        lab = ""
        if k in SOLO_TESTS:
            lab = SOLO_TESTS[k][0]
        elif k in combo_tests:
            lab = combo_tests[k][0]
        else:
            lab = k

        dc = (r["cagr"] - base["cagr"]) * 100
        print(f"\n--- {lab} ({k}) ---")
        print(f"  CAGR:        {r['cagr']*100:.2f}% ({dc:+.2f}pp vs baseline)")
        print(f"  Final Value: Rs {r['final_value']/1e7:.2f} Cr ({r['final_value']/config.initial_capital:.1f}x)")
        print(f"  Max DD:      {r['max_dd']*100:.2f}%")
        print(f"  Longest DD:  {r['longest_dd']}d")
        print(f"  Volatility:  {r['vol']*100:.2f}%")
        print(f"  Sharpe:      {r['sharpe']:.2f}")
        print(f"  Sortino:     {r['sortino']:.2f}")
        print(f"  Calmar:      {r['calmar']:.2f}")
        print(f"  Win Rate:    {r['win_rate']*100:.1f}%")
        print(f"  PF:          {r['profit_factor']:.2f}")
        print(f"  Avg Hold:    {r['avg_hold']:.1f}d")
        print(f"  Avg Pos:     {r['avg_positions']:.1f}")
        print(f"  Avg Exp:     {r['avg_exposure']*100:.1f}%")
        print(f"  Avg Turn:    {r['avg_turnover']*100:.2f}%")
        print(f"  Best Year:   {r['best_year']*100:.1f}%")
        print(f"  Worst Year:  {r['worst_year']*100:.1f}%")
        print(f"  +ve Years:   {r['pos_years']}/{r['total_years']}")

        if r["rolling_3yr"]:
            r3s = pd.Series(r["rolling_3yr"])
            print(f"  Roll 3yr CAGR: median={r3s.median()*100:.1f}% min={r3s.min()*100:.1f}% "
                  f">15%={((r3s>0.15).mean()*100):.0f}% neg={(r3s<0).mean()*100:.0f}%")

    total = time.time() - t0
    print(f"\n\nTotal runtime: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
