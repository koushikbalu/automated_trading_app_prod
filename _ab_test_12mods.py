"""A/B test harness: 12 proposed modifications vs current baseline.

Downloads data ONCE, then runs 13 backtests (1 baseline + 12 mods)
using the same market data. Each modification patches only the specific
logic it changes; everything else uses baseline config.yaml settings.

Usage:  python _ab_test_12mods.py
"""
from __future__ import annotations

import copy
import logging
import sys
import io
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("ab_test")
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

# ─────────────────────────────────────────────────────────────────────
# Core backtest engine (copied from backtest.py run_backtest, with
# hook points for modifications)
# ─────────────────────────────────────────────────────────────────────

def run_backtest_with_mods(
    config: Config,
    raw_data: dict[str, pd.DataFrame],
    *,
    mod_id: str = "baseline",
) -> dict:
    """Run the full backtest with per-modification hooks injected
    based on mod_id. Data is passed in (no re-download)."""

    sector_map = _yf_sector_map(SECTOR_MAP)

    close = raw_data["Adj Close"].copy()
    high = raw_data["High"].copy()
    low = raw_data["Low"].copy()
    volume = raw_data["Volume"].copy()

    has_mom_bench = (
        config.momentum_benchmark
        and config.momentum_benchmark in close.columns
        and close[config.momentum_benchmark].dropna().shape[0] > 252
    )

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
    stocks_volume = (
        volume.drop(columns=[config.benchmark], errors="ignore").reindex(benchmark.index).fillna(0)
    )

    daily_returns = stocks_close.pct_change().fillna(0)
    benchmark_returns = benchmark.pct_change().fillna(0)

    if has_mom_bench:
        mom_bench_series = close[config.momentum_benchmark].dropna().reindex(benchmark.index).ffill()
        mom_bench_returns = mom_bench_series.pct_change().fillna(0)
    else:
        mom_bench_returns = pd.Series(0.0, index=benchmark.index)

    traded_value = stocks_close * stocks_volume
    adv_126 = traded_value.rolling(126).mean()
    dma_20 = stocks_close.rolling(20).mean()
    dma_50 = stocks_close.rolling(50).mean()
    dma_100 = stocks_close.rolling(100).mean()
    dma_200 = stocks_close.rolling(200).mean()
    vol_60 = daily_returns.rolling(60).std(ddof=1) * np.sqrt(252)

    atr = compute_atr_df(stocks_high, stocks_low, stocks_close, config.atr_window)

    bench_50 = benchmark.rolling(50).mean()
    bench_200 = benchmark.rolling(200).mean()
    bench_63ret = benchmark / benchmark.shift(63) - 1

    # Benchmark rolling vol for mod1 (adaptive momentum)
    bench_vol_60 = benchmark_returns.rolling(60).std(ddof=1) * np.sqrt(252)

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

    # Mod9: staggered rebalance -- build T-1, T-2 sets
    if mod_id == "mod09_staggered_rebal":
        _rebal_t_minus_1 = _month_groups.apply(
            lambda g: g.iloc[-2] if len(g) >= 2 else g.iloc[-1]
        )
        _rebal_t_minus_2 = _month_groups.apply(
            lambda g: g.iloc[-3] if len(g) >= 3 else g.iloc[-1]
        )
        stagger_sell_dates = set(_rebal_t_minus_2.values)
        stagger_buy1_dates = set(_rebal_t_minus_1.values)
        stagger_buy2_dates = rebalance_dates  # T day: buy remaining 50%

    first_trading_day_of_month = pd.Series(
        benchmark.index, index=benchmark.index
    ).groupby([benchmark.index.year, benchmark.index.month]).first()
    first_td_set = set(first_trading_day_of_month.values)

    # Pre-compute correlation matrix for mod8 (one per rebalance, not per day)
    # Will be computed lazily at rebalance time

    # State
    portfolio_value = config.initial_capital
    benchmark_value = config.initial_capital
    mom_bench_value = config.initial_capital
    daily_cash_ret = (1 + config.annual_cash_yield) ** (1 / 252) - 1

    positions: dict[str, dict] = {}
    pending_exits: list[str] = []
    stopped_out_this_month: dict[str, float] = {}
    last_rebal_cand: pd.DataFrame = pd.DataFrame()
    last_rebal_month: tuple[int, int] = (0, 0)
    portfolio_peak = config.initial_capital
    circuit_breaker_active = False
    cb_risk_on_streak = 0

    # Mod9 staggered state
    stagger_pending_sells: list[str] = []
    stagger_pending_buys_half: dict[str, float] = {}  # ticker -> weight (50% to buy on T)

    trades_records: list[dict] = []
    daily_records: list[dict] = []

    mw = config.momentum_weights
    start_index = max(252, 200)

    for i in range(start_index, len(benchmark.index)):
        current_date = benchmark.index[i]
        buy_turnover = 0.0
        sell_turnover = 0.0

        if (
            current_date in first_td_set
            and current_date.month == config.addition_month
            and config.annual_addition > 0
        ):
            portfolio_value += config.annual_addition
            benchmark_value += config.annual_addition
            mom_bench_value += config.annual_addition

        benchmark_value *= 1 + benchmark_returns.loc[current_date]
        mom_bench_value *= 1 + mom_bench_returns.loc[current_date]

        # ── Regime detection ──
        above_200 = stocks_close.loc[current_date] > dma_200.loc[current_date]
        notna_count = above_200.notna().sum()
        breadth = above_200.sum() / notna_count if notna_count > 0 else np.nan

        bench_above_200 = (
            pd.notna(bench_200.loc[current_date])
            and benchmark.loc[current_date] > bench_200.loc[current_date]
        )
        full_risk_on = (
            bench_above_200
            and (bench_50.loc[current_date] > bench_200.loc[current_date])
            and (not config.require_positive_3m_return or bench_63ret.loc[current_date] > 0)
            and (breadth > config.breadth_threshold)
        )
        neutral = (
            not full_risk_on
            and bench_above_200
            and (breadth > config.neutral_breadth_threshold)
        )

        if full_risk_on:
            allocation_pct = 1.0
        elif neutral:
            allocation_pct = config.neutral_allocation_pct
        else:
            allocation_pct = 0.0

        # ── Mod7: Gradient drawdown allocation (replaces binary CB) ──
        portfolio_peak = max(portfolio_peak, portfolio_value)
        portfolio_dd = (portfolio_value / portfolio_peak) - 1 if portfolio_peak > 0 else 0.0

        if mod_id == "mod07_gradient_dd":
            if portfolio_dd < -0.05:
                dd_scale = max(0.0, 1.0 + (portfolio_dd + 0.05) / 0.15)
                allocation_pct *= dd_scale
            if portfolio_dd < -0.20:
                for ticker in list(positions.keys()):
                    sell_turnover += positions[ticker]["weight"]
                    del positions[ticker]
        else:
            # Standard CB logic
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
                    del positions[ticker]

        # ── Process pending exits (T+1) ──
        for ticker in pending_exits:
            if ticker in positions:
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]
        pending_exits = []

        # ── Force-exit extended missing data ──
        for ticker in list(positions.keys()):
            if _missing_streak.loc[current_date, ticker] > config.ffill_limit:
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]

        cur_ym = (current_date.year, current_date.month)
        if cur_ym != last_rebal_month and last_rebal_month != (0, 0):
            stopped_out_this_month = {}

        # ── Monthly rebalance ──
        do_rebalance = current_date in rebalance_dates

        # Mod10: skip-month if overlap > 80%
        if mod_id == "mod10_skip_low_turnover" and do_rebalance and positions:
            # Preview what the new portfolio would be
            pass  # check after candidate selection below

        if do_rebalance:
            stopped_out_this_month = {}
            last_rebal_month = cur_ym

            liquid = adv_126.loc[current_date].dropna().sort_values(ascending=False)
            liquid = liquid[stocks_close.loc[current_date, liquid.index] >= config.min_price]
            liquid_universe = liquid.head(config.top_liquid_n).index.tolist()
            num_slots = int(round(config.top_momentum_n * allocation_pct))

            if num_slots > 0 and len(liquid_universe) > 0:
                lu = liquid_universe

                # ── Mod1: Adaptive momentum lookback ──
                if mod_id == "mod01_adaptive_lookback":
                    bv = bench_vol_60.loc[current_date]
                    if pd.notna(bv) and bv > 0.22:
                        active_mw = [0.10, 0.25, 0.35, 0.30]
                    else:
                        active_mw = [0.30, 0.40, 0.20, 0.10]
                else:
                    active_mw = mw

                ret_1m = stocks_close.loc[current_date, lu] / stocks_close.shift(21).loc[current_date, lu] - 1
                ret_3m = stocks_close.loc[current_date, lu] / stocks_close.shift(63).loc[current_date, lu] - 1
                ret_6m = stocks_close.loc[current_date, lu] / stocks_close.shift(126).loc[current_date, lu] - 1
                ret_12m = stocks_close.shift(21).loc[current_date, lu] / stocks_close.shift(252).loc[current_date, lu] - 1

                raw_mom = active_mw[0] * ret_12m + active_mw[1] * ret_6m + active_mw[2] * ret_3m + active_mw[3] * ret_1m

                elig = (
                    (stocks_close.loc[current_date, lu] > dma_100.loc[current_date, lu])
                    & (stocks_close.loc[current_date, lu] > dma_200.loc[current_date, lu])
                )

                # ── Mod2: Absolute momentum overlay ──
                if mod_id == "mod02_abs_momentum":
                    abs_mom_ok = ret_12m > 0
                    elig = elig & abs_mom_ok.reindex(elig.index, fill_value=False)

                vol_20 = stocks_volume.iloc[max(0, i - 20): i + 1].mean()
                vol_60_avg = stocks_volume.iloc[max(0, i - 60): i + 1].mean()
                rel_volume = (vol_20 / vol_60_avg.replace(0, np.nan)).reindex(lu)
                volume_ok = rel_volume >= rel_volume.median()

                cand = pd.DataFrame({
                    "score": raw_mom,
                    "vol": vol_60.loc[current_date, lu],
                    "eligible": elig,
                    "volume_ok": volume_ok,
                }).dropna()

                cand = cand[cand["eligible"] & cand["volume_ok"]]
                if config.max_vol_percentile < 1.0 and not cand.empty:
                    vol_cap = cand["vol"].quantile(config.max_vol_percentile)
                    cand = cand[cand["vol"] <= vol_cap]
                cand["risk_adj_score"] = cand["score"] / cand["vol"].replace(0, np.nan)
                cand = cand.dropna()

                # ── Mod3: Acceleration filter ──
                if mod_id == "mod03_acceleration" and not cand.empty:
                    ret_3m_prev = stocks_close.shift(21).loc[current_date, cand.index] / stocks_close.shift(84).loc[current_date, cand.index] - 1
                    accel = ret_3m.reindex(cand.index) - ret_3m_prev
                    accel = accel.reindex(cand.index)
                    cand.loc[accel > 0, "risk_adj_score"] *= 1.15

                # ── Mod5: Earnings momentum proxy (price reaction) ──
                if mod_id == "mod05_earnings_proxy" and not cand.empty:
                    ret_5d = stocks_close.loc[current_date, cand.index] / stocks_close.shift(5).loc[current_date, cand.index] - 1
                    ret_5d_prev_q = stocks_close.shift(63).loc[current_date, cand.index] / stocks_close.shift(68).loc[current_date, cand.index] - 1
                    both_positive = (ret_5d > 0.02) & (ret_5d_prev_q > 0.02)
                    cand.loc[both_positive.reindex(cand.index, fill_value=False), "risk_adj_score"] *= 1.10

                if config.sector_downtrend_penalty < 1.0 and not cand.empty:
                    cand["_sector"] = cand.index.map(lambda t: sector_map.get(t, "Other"))
                    sector_median_mom = cand.groupby("_sector")["score"].median()
                    bad_sectors = set(sector_median_mom[sector_median_mom < 0].index)
                    if bad_sectors:
                        mask = cand["_sector"].isin(bad_sectors)
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

                # ── Mod8: Correlation-aware selection ──
                if mod_id == "mod08_corr_aware" and not cand.empty and len(cand) > num_slots:
                    corr_window = daily_returns.iloc[max(0, i - 60): i + 1]
                    pre_selected: list[str] = []
                    for t in cand.index:
                        sec = sector_map.get(t, "Other")
                        if sector_counts.get(sec, 0) >= max_per_sector:
                            continue
                        if not pre_selected:
                            pre_selected.append(t)
                            sector_counts[sec] = sector_counts.get(sec, 0) + 1
                            continue
                        # Check avg correlation with already selected
                        if t in corr_window.columns and len(pre_selected) > 0:
                            try:
                                selected_rets = corr_window[pre_selected]
                                new_ret = corr_window[t]
                                avg_corr = selected_rets.corrwith(new_ret).mean()
                                if pd.notna(avg_corr) and avg_corr > 0.65:
                                    continue  # too correlated, skip
                            except Exception:
                                pass
                        pre_selected.append(t)
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1
                        if len(pre_selected) >= num_slots:
                            break
                    filtered_idx = pre_selected
                else:
                    for t in cand.index:
                        sec = sector_map.get(t, "Other")
                        if sector_counts.get(sec, 0) < max_per_sector:
                            filtered_idx.append(t)
                            sector_counts[sec] = sector_counts.get(sec, 0) + 1
                        if len(filtered_idx) >= num_slots:
                            break

                selected = filtered_idx

                # ── Mod10: skip if overlap > 80% ──
                if mod_id == "mod10_skip_low_turnover" and positions and selected:
                    overlap = len(set(positions.keys()) & set(selected)) / len(selected) if selected else 0
                    if overlap > 0.80:
                        # Skip full rebalance, only adjust the changed positions
                        new_set = set(selected)
                        old_set = set(positions.keys())
                        to_exit = old_set - new_set
                        for ticker in to_exit:
                            sell_turnover += positions[ticker]["weight"]
                            del positions[ticker]
                        # Don't do full re-weighting, keep existing weights
                        # Just add any new entries with equal weight
                        to_add = new_set - old_set
                        if to_add:
                            add_w = 0.05  # small initial weight
                            for ticker in to_add:
                                price = stocks_close.loc[current_date, ticker]
                                atr_val = atr.loc[current_date, ticker]
                                stop_price = None
                                if config.use_atr_trailing_stop and pd.notna(atr_val):
                                    stop_price = float(price - config.atr_multiple * atr_val)
                                positions[ticker] = {
                                    "weight": add_w,
                                    "entry_price": float(price),
                                    "entry_date": current_date,
                                    "high_watermark": float(price),
                                    "stop_price": stop_price,
                                    "losing_days": 0,
                                    "pyramid_count": 0,
                                }
                                buy_turnover += add_w
                        selected = []  # skip rest of rebalance logic

                if selected:
                    # ── Mod11: Dynamic max weight by rank ──
                    if mod_id == "mod11_dynamic_max_weight":
                        per_stock_caps = {}
                        for rank_i, t in enumerate(selected):
                            if rank_i < 3:
                                per_stock_caps[t] = 0.22
                            elif rank_i < 7:
                                per_stock_caps[t] = 0.15
                            else:
                                per_stock_caps[t] = 0.10
                        # Use per-stock caps in blended weights
                        max_cap = max(per_stock_caps.values())
                        new_weights = blended_weights(
                            cand.loc[selected, "vol"],
                            cand.loc[selected, "risk_adj_score"],
                            max_cap,
                            config.score_blend,
                        )
                        for t in selected:
                            if new_weights.get(t, 0) > per_stock_caps.get(t, config.max_weight):
                                new_weights[t] = per_stock_caps[t]
                        if new_weights.sum() > 0:
                            new_weights = new_weights / new_weights.sum()
                    else:
                        new_weights = blended_weights(
                            cand.loc[selected, "vol"],
                            cand.loc[selected, "risk_adj_score"],
                            config.max_weight,
                            config.score_blend,
                        )

                    keep = new_weights[new_weights >= config.min_weight]
                    if not keep.empty:
                        new_weights = keep / keep.sum()
                        selected = list(keep.index)
                    else:
                        new_weights = pd.Series(dtype=float)
                        selected = []

                    # ── Mod6: Volatility-targeted sizing ──
                    if mod_id == "mod06_vol_target" and selected:
                        target_vol = 0.15
                        port_var = 0.0
                        for t in selected:
                            w = new_weights.get(t, 0)
                            v = vol_60.loc[current_date, t] if t in vol_60.columns and pd.notna(vol_60.loc[current_date, t]) else 0.20
                            port_var += (w * v) ** 2
                        port_vol_est = np.sqrt(port_var)
                        if port_vol_est > target_vol and port_vol_est > 0:
                            scale = target_vol / port_vol_est
                            new_weights = new_weights * scale
                else:
                    new_weights = pd.Series(dtype=float)
            else:
                liquid_universe = []
                selected = []
                new_weights = pd.Series(dtype=float)

            existing = set(positions.keys())
            target = set(selected)

            for ticker in list(existing - target):
                sell_turnover += positions.get(ticker, {}).get("weight", 0)
                if ticker in positions:
                    del positions[ticker]

            for ticker in selected:
                price = stocks_close.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]
                new_w = float(new_weights.get(ticker, 0))

                if ticker not in positions:
                    stop_price = None
                    if config.use_atr_trailing_stop and pd.notna(atr_val):
                        stop_price = float(price - config.atr_multiple * atr_val)
                    positions[ticker] = {
                        "weight": new_w,
                        "entry_price": float(price),
                        "entry_date": current_date,
                        "high_watermark": float(price),
                        "stop_price": stop_price,
                        "losing_days": 0,
                        "pyramid_count": 0,
                    }
                    buy_turnover += new_w
                else:
                    old_w = positions[ticker]["weight"]
                    delta = new_w - old_w
                    positions[ticker]["weight"] = new_w
                    if delta > 0:
                        buy_turnover += delta
                    else:
                        sell_turnover += abs(delta)

        # ── Daily re-entry ──
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
                        "weight": orig_w,
                        "entry_price": float(price),
                        "entry_date": current_date,
                        "high_watermark": float(price),
                        "stop_price": stop_price,
                        "losing_days": 0,
                        "pyramid_count": 0,
                    }
                    buy_turnover += orig_w

        # ── Mid-month pyramiding ──
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

        # ── Min exposure floor ──
        if not last_rebal_cand.empty and allocation_pct > 0:
            pos_exposure = sum(p["weight"] for p in positions.values())
            if pos_exposure < config.min_exposure:
                needed = config.min_exposure - pos_exposure
                already_held = set(positions.keys()) | set(pending_exits)
                floor_candidates = [
                    t for t in last_rebal_cand.index
                    if t not in already_held
                    and pd.notna(stocks_close.loc[current_date, t])
                    and stocks_close.loc[current_date, t] >= config.min_price
                ][:config.min_exposure_slots]
                if floor_candidates:
                    floor_vols = last_rebal_cand.loc[floor_candidates, "vol"]
                    floor_weights = capped_inverse_vol_weights(floor_vols, config.max_weight)
                    floor_weights = floor_weights / floor_weights.sum() * needed
                    for ticker in floor_candidates:
                        fw = float(floor_weights.loc[ticker])
                        price = stocks_close.loc[current_date, ticker]
                        atr_val = atr.loc[current_date, ticker]
                        stop_price = None
                        if config.use_atr_trailing_stop and pd.notna(atr_val):
                            stop_price = float(price - config.atr_multiple * atr_val)
                        positions[ticker] = {
                            "weight": fw,
                            "entry_price": float(price),
                            "entry_date": current_date,
                            "high_watermark": float(price),
                            "stop_price": stop_price,
                            "losing_days": 0,
                            "pyramid_count": 0,
                        }
                        buy_turnover += fw

        # ── Daily exit signals + PnL ──
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
                    # ── Mod4: Maturing stop tightening ──
                    if mod_id == "mod04_maturing_stop" and "entry_date" in positions[ticker]:
                        days_held = (current_date - positions[ticker]["entry_date"]).days
                        if days_held < 10:
                            eff_mult = config.atr_multiple * 1.2
                        elif days_held > 60:
                            eff_mult = config.atr_multiple * 0.75
                        else:
                            eff_mult = config.atr_multiple
                    else:
                        eff_mult = config.atr_multiple

                    candidate_stop = positions[ticker]["high_watermark"] - eff_mult * atr_val
                    if positions[ticker]["stop_price"] is None:
                        positions[ticker]["stop_price"] = float(candidate_stop)
                    else:
                        positions[ticker]["stop_price"] = max(positions[ticker]["stop_price"], float(candidate_stop))

                exit_reason = None
                if (
                    config.use_atr_trailing_stop
                    and positions[ticker]["stop_price"] is not None
                    and price < positions[ticker]["stop_price"]
                ):
                    exit_reason = "ATR Trailing Stop"

                if exit_reason is None and config.max_loss_pct is not None:
                    pnl_pct = (price / positions[ticker]["entry_price"]) - 1
                    if pnl_pct < config.max_loss_pct:
                        exit_reason = "Hard stop"

                if price < positions[ticker]["entry_price"]:
                    positions[ticker]["losing_days"] += 1
                else:
                    positions[ticker]["losing_days"] = 0

                if config.max_hold_losing_days > 0 and exit_reason is None and positions[ticker]["losing_days"] >= config.max_hold_losing_days:
                    exit_reason = "Time stop"

                if exit_reason is not None:
                    stopped_out_this_month[ticker] = positions[ticker]["weight"]
                    pending_exits.append(ticker)
                    trades_records.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": float(price * (1 - config.stop_exit_slippage)),
                        "weight_traded": positions[ticker]["weight"],
                        "reason": exit_reason,
                    })

            pre_move_total = 0.0
            for ticker in positions:
                if "current_value" not in positions[ticker]:
                    positions[ticker]["current_value"] = portfolio_value * positions[ticker]["weight"]
                pre_move_total += positions[ticker]["current_value"]

            invested_frac = pre_move_total / portfolio_value if portfolio_value > 0 else 0.0

            for ticker in positions:
                dr_val = daily_returns.loc[current_date, ticker]
                w_frac = positions[ticker]["current_value"] / portfolio_value if portfolio_value > 0 else 0.0
                gross_ret += w_frac * dr_val
                positions[ticker]["current_value"] *= 1 + dr_val

            if invested_frac < 1:
                gross_ret += (1 - invested_frac) * daily_cash_ret

            post_move_total = sum(positions[t]["current_value"] for t in positions)
            if post_move_total > 0:
                for ticker in positions:
                    positions[ticker]["weight"] = positions[ticker]["current_value"] / post_move_total

        trading_cost = buy_turnover * config.buy_side_cost + sell_turnover * config.sell_side_cost

        # ── Mod12: Hedge cost proxy ──
        if mod_id == "mod12_tail_hedge":
            trading_cost += 0.008 / 21  # ~1% per month / 21 trading days

        net_ret = gross_ret - trading_cost
        portfolio_value *= 1 + net_ret

        if positions:
            total_w = sum(positions[t]["weight"] for t in positions)
            for t in positions:
                positions[t]["current_value"] = portfolio_value * (positions[t]["weight"] / total_w if total_w > 0 else 0)

        pos_weight_sum = sum(positions[t]["weight"] for t in positions) if positions else 0.0
        daily_records.append({
            "date": current_date,
            "strategy_daily_return": net_ret,
            "benchmark_daily_return": benchmark_returns.loc[current_date],
            "strategy_value": portfolio_value,
            "benchmark_value": benchmark_value,
            "turnover": buy_turnover + sell_turnover,
            "positions_count": len(positions),
            "cash_weight": max(0.0, 1 - pos_weight_sum) if positions else 1.0,
        })

    # ── Compute summary metrics ──
    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    trades_df = pd.DataFrame(trades_records)

    strategy_curve = daily_df.set_index("date")["strategy_value"]
    benchmark_curve = daily_df.set_index("date")["benchmark_value"]
    strategy_daily = daily_df.set_index("date")["strategy_daily_return"]
    benchmark_daily = daily_df.set_index("date")["benchmark_daily_return"]

    strat_cagr = annualized_return(strategy_curve)
    bench_cagr = annualized_return(benchmark_curve)
    strat_mdd = max_drawdown(strategy_curve)
    strat_vol = annualized_vol(strategy_daily)
    strat_sharpe = sharpe_ratio(strategy_daily, config.annual_cash_yield)

    daily_rf = (1 + config.annual_cash_yield) ** (1 / 252) - 1
    strat_downside = strategy_daily[strategy_daily < 0].std(ddof=1) * np.sqrt(252)
    strat_sortino = (
        (strategy_daily - daily_rf).mean()
        / strategy_daily[strategy_daily < 0].std(ddof=1)
        * np.sqrt(252)
        if strat_downside > 0 else np.nan
    )
    strat_calmar = strat_cagr / abs(strat_mdd) if strat_mdd != 0 else np.nan

    strat_peak = strategy_curve.cummax()
    strat_in_dd = strategy_curve < strat_peak
    dd_groups = (~strat_in_dd).cumsum()
    dd_durations = strat_in_dd.groupby(dd_groups).sum()
    longest_dd_days = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Roundtrip trades
    roundtrip_rows: list[dict] = []
    if not trades_df.empty:
        open_rt: dict[str, dict] = {}
        for _, row in trades_df.sort_values("date").iterrows():
            ticker = row["ticker"]
            action = row["action"]
            price = row["price"]
            weight = row["weight_traded"]
            dt = row["date"]
            if action == "BUY":
                if ticker not in open_rt:
                    open_rt[ticker] = {"total_cost": price * weight, "total_weight": weight, "entry_date": dt, "pyramids": 0}
                else:
                    p = open_rt[ticker]
                    p["total_cost"] += price * weight
                    p["total_weight"] += weight
                    p["pyramids"] += 1
            elif action in ("SELL", "REBAL_EXIT"):
                if ticker in open_rt:
                    p = open_rt.pop(ticker)
                    wavg = p["total_cost"] / p["total_weight"]
                    ret = price / wavg - 1 if wavg > 0 else np.nan
                    hold = (pd.Timestamp(dt) - pd.Timestamp(p["entry_date"])).days
                    roundtrip_rows.append({"return": ret, "holding_days": hold})

    if roundtrip_rows:
        rt_df = pd.DataFrame(roundtrip_rows)
        trade_returns = rt_df["return"].dropna()
        win_rate = (trade_returns > 0).mean()
        avg_winner = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
        avg_loser = trade_returns[trade_returns <= 0].mean() if (trade_returns <= 0).any() else 0
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns <= 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_hold = rt_df["holding_days"].mean()
        total_rt = len(rt_df)
    else:
        win_rate = avg_winner = avg_loser = profit_factor = avg_hold = np.nan
        total_rt = 0

    avg_turnover = daily_df["turnover"].mean()
    avg_positions = daily_df["positions_count"].mean()
    avg_exposure = (1 - daily_df["cash_weight"]).mean()

    return {
        "cagr": strat_cagr,
        "max_dd": strat_mdd,
        "longest_dd": longest_dd_days,
        "vol": strat_vol,
        "sharpe": strat_sharpe,
        "sortino": strat_sortino,
        "calmar": strat_calmar,
        "win_rate": win_rate,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "total_roundtrips": total_rt,
        "avg_hold": avg_hold,
        "avg_turnover": avg_turnover,
        "avg_positions": avg_positions,
        "avg_exposure": avg_exposure,
        "final_value": strategy_curve.iloc[-1],
        "bench_cagr": bench_cagr,
    }


# ─────────────────────────────────────────────────────────────────────
# Test definitions
# ─────────────────────────────────────────────────────────────────────

TESTS = {
    "00_baseline":          ("Baseline (current config.yaml)", "baseline", {}),
    "01_adaptive_lookback": ("Adaptive Momentum Lookback", "mod01_adaptive_lookback", {}),
    "02_abs_momentum":      ("Absolute Momentum Overlay", "mod02_abs_momentum", {}),
    "03_acceleration":      ("Acceleration Filter (+15% boost)", "mod03_acceleration", {}),
    "04_maturing_stop":     ("Maturing Stop Tightening", "mod04_maturing_stop", {}),
    "05_earnings_proxy":    ("Earnings Momentum Proxy", "mod05_earnings_proxy", {}),
    "06_vol_target":        ("Vol-Targeted Sizing (15%)", "mod06_vol_target", {}),
    "07_gradient_dd":       ("Gradient Drawdown Alloc", "mod07_gradient_dd", {}),
    "08_corr_aware":        ("Correlation-Aware Selection", "mod08_corr_aware", {}),
    "09_staggered_rebal":   ("Staggered Rebalance (3-day)", "mod09_staggered_rebal", {}),
    "10_skip_low_turnover": ("Skip-Month Low Turnover", "mod10_skip_low_turnover", {}),
    "11_dynamic_max_wt":    ("Dynamic Max Weight by Rank", "mod11_dynamic_max_weight", {}),
    "12_tail_hedge":        ("Tail Hedge Cost Proxy (1%/mo)", "mod12_tail_hedge", {}),
}


def main():
    t0 = time.time()

    # Load config from YAML (the current production config)
    config = load_config_from_yaml()
    config = replace(config, start_date="2016-04-01")
    logger.info("Config loaded. Start date: %s", config.start_date)

    # Download data ONCE
    yf_universe = _yf_tickers(BROAD_UNIVERSE)
    extra = [config.benchmark]
    if config.momentum_benchmark:
        extra.append(config.momentum_benchmark)
    tickers = yf_universe + extra

    logger.info("Downloading data for %d tickers (this takes a few minutes)...", len(tickers))
    from backtest import download_ohlcv
    raw_data = download_ohlcv(tickers, config.start_date, config.end_date)
    dl_time = time.time() - t0
    logger.info("Data downloaded in %.0fs. Running 13 backtests...", dl_time)

    # Run all tests
    results: dict[str, dict] = {}
    for test_key, (label, mod_id, cfg_overrides) in TESTS.items():
        t1 = time.time()
        test_config = replace(config, **cfg_overrides) if cfg_overrides else config
        logger.info("Running [%s] %s ...", test_key, label)
        try:
            res = run_backtest_with_mods(test_config, raw_data, mod_id=mod_id)
            results[test_key] = res
            logger.info(
                "  => CAGR=%.2f%% MaxDD=%.2f%% Sharpe=%.2f (%.0fs)",
                res["cagr"] * 100, res["max_dd"] * 100, res["sharpe"], time.time() - t1,
            )
        except Exception as exc:
            logger.error("  => FAILED: %s", exc)
            import traceback; traceback.print_exc()

    # ── Print comparison table ──
    print("\n" + "=" * 130)
    print("A/B TEST RESULTS -- 12 Modifications vs Baseline")
    print("=" * 130)

    base = results.get("00_baseline")
    if not base:
        print("ERROR: Baseline failed. Cannot compare.")
        return

    header = f"{'#':<4} {'Modification':<35} {'CAGR':>7} {'dCAGR':>7} {'MaxDD':>8} {'dDD':>7} {'Sharpe':>7} {'dShp':>6} {'Sortino':>8} {'Calmar':>7} {'WinR':>6} {'PF':>6} {'AvgTurn':>8}"
    print(header)
    print("-" * 130)

    for test_key, (label, _, _) in TESTS.items():
        r = results.get(test_key)
        if not r:
            print(f"{'--':<4} {label:<35} {'FAILED':>7}")
            continue

        d_cagr = (r["cagr"] - base["cagr"]) * 100
        d_dd = (r["max_dd"] - base["max_dd"]) * 100
        d_sharpe = r["sharpe"] - base["sharpe"]

        marker = ""
        if test_key != "00_baseline":
            if d_cagr > 0.5 and d_sharpe >= 0:
                marker = " [WIN]"
            elif d_cagr < -0.5:
                marker = " [LOSE]"

        print(
            f"{test_key[:4]:<4} {label:<35} "
            f"{r['cagr']*100:>6.2f}% {d_cagr:>+6.2f} "
            f"{r['max_dd']*100:>7.2f}% {d_dd:>+6.2f} "
            f"{r['sharpe']:>6.2f} {d_sharpe:>+5.2f} "
            f"{r['sortino']:>7.2f} {r['calmar']:>6.2f} "
            f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
            f"{r['avg_turnover']*100:>7.2f}%{marker}"
        )

    print("-" * 130)

    # Detailed per-test analysis
    print("\n\n" + "=" * 100)
    print("DETAILED PER-TEST ANALYSIS")
    print("=" * 100)

    for test_key, (label, _, _) in TESTS.items():
        if test_key == "00_baseline":
            continue
        r = results.get(test_key)
        if not r:
            continue

        d_cagr = (r["cagr"] - base["cagr"]) * 100
        d_dd = (r["max_dd"] - base["max_dd"]) * 100
        d_sharpe = r["sharpe"] - base["sharpe"]
        d_sortino = r["sortino"] - base["sortino"]

        print(f"\n--- {label} ({test_key}) ---")
        print(f"  CAGR:       {r['cagr']*100:.2f}% ({d_cagr:+.2f}pp vs baseline)")
        print(f"  Max DD:     {r['max_dd']*100:.2f}% ({d_dd:+.2f}pp)")
        print(f"  Longest DD: {r['longest_dd']}d (baseline: {base['longest_dd']}d)")
        print(f"  Volatility: {r['vol']*100:.2f}% (baseline: {base['vol']*100:.2f}%)")
        print(f"  Sharpe:     {r['sharpe']:.2f} ({d_sharpe:+.2f})")
        print(f"  Sortino:    {r['sortino']:.2f} ({d_sortino:+.2f})")
        print(f"  Calmar:     {r['calmar']:.2f} (baseline: {base['calmar']:.2f})")
        print(f"  Win Rate:   {r['win_rate']*100:.1f}% (baseline: {base['win_rate']*100:.1f}%)")
        print(f"  PF:         {r['profit_factor']:.2f} (baseline: {base['profit_factor']:.2f})")
        print(f"  Avg Winner: {r['avg_winner']*100:.2f}% (baseline: {base['avg_winner']*100:.2f}%)")
        print(f"  Avg Loser:  {r['avg_loser']*100:.2f}% (baseline: {base['avg_loser']*100:.2f}%)")
        print(f"  Avg Hold:   {r['avg_hold']:.1f}d (baseline: {base['avg_hold']:.1f}d)")
        print(f"  Avg Turn:   {r['avg_turnover']*100:.2f}% (baseline: {base['avg_turnover']*100:.2f}%)")
        print(f"  Avg Pos:    {r['avg_positions']:.1f} (baseline: {base['avg_positions']:.1f})")
        print(f"  Final Val:  {r['final_value']:,.0f} (baseline: {base['final_value']:,.0f})")

        verdict = "WINNER" if d_cagr > 0.5 and d_sharpe >= 0 else "NEUTRAL" if abs(d_cagr) <= 0.5 else "LOSER"
        print(f"  VERDICT:    {verdict}")

    total_time = time.time() - t0
    print(f"\n\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
