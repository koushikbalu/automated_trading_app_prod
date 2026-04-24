"""20-year backtest: Baseline vs Mod 07 (Gradient Drawdown Allocation).

Downloads data ONCE from 2006-04-01, runs both backtests, and produces
a detailed year-by-year, drawdown, and risk analysis report.

Usage:  python _bt_20yr_mod07.py
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
logger = logging.getLogger("bt20yr")
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


def run_backtest_core(config: Config, raw_data: dict, *, use_gradient_dd: bool = False) -> dict:
    """Run full backtest, returning daily equity curve + detailed stats.
    If use_gradient_dd=True, replaces binary circuit breaker with gradient."""

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
    stocks_volume = volume.drop(columns=[config.benchmark], errors="ignore").reindex(benchmark.index).fillna(0)

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

        # Regime detection
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

        portfolio_peak = max(portfolio_peak, portfolio_value)
        portfolio_dd = (portfolio_value / portfolio_peak) - 1 if portfolio_peak > 0 else 0.0

        if use_gradient_dd:
            if portfolio_dd < -0.05:
                dd_scale = max(0.0, 1.0 + (portfolio_dd + 0.05) / 0.15)
                allocation_pct *= dd_scale
            if portfolio_dd < -0.20:
                for ticker in list(positions.keys()):
                    sell_turnover += positions[ticker]["weight"]
                    trades_records.append({
                        "date": current_date, "ticker": ticker, "action": "SELL",
                        "price": stocks_close.loc[current_date, ticker],
                        "weight_traded": positions[ticker]["weight"], "reason": "Gradient CB liquidation",
                    })
                    del positions[ticker]
        else:
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
                    trades_records.append({
                        "date": current_date, "ticker": ticker, "action": "SELL",
                        "price": stocks_close.loc[current_date, ticker],
                        "weight_traded": positions[ticker]["weight"], "reason": "CB liquidation",
                    })
                    del positions[ticker]

        for ticker in pending_exits:
            if ticker in positions:
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]
        pending_exits = []

        for ticker in list(positions.keys()):
            if _missing_streak.loc[current_date, ticker] > config.ffill_limit:
                sell_turnover += positions[ticker]["weight"]
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
                elig = (
                    (stocks_close.loc[current_date, lu] > dma_100.loc[current_date, lu])
                    & (stocks_close.loc[current_date, lu] > dma_200.loc[current_date, lu])
                )

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
                liquid_universe = []
                selected = []
                new_weights = pd.Series(dtype=float)

            existing = set(positions.keys())
            target = set(selected)

            for ticker in list(existing - target):
                trades_records.append({
                    "date": current_date, "ticker": ticker, "action": "SELL",
                    "price": stocks_close.loc[current_date, ticker],
                    "weight_traded": positions.get(ticker, {}).get("weight", 0),
                    "reason": "Rebal exit",
                })
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
                        "weight": new_w, "entry_price": float(price),
                        "entry_date": current_date, "high_watermark": float(price),
                        "stop_price": stop_price, "losing_days": 0, "pyramid_count": 0,
                    }
                    buy_turnover += new_w
                    trades_records.append({
                        "date": current_date, "ticker": ticker, "action": "BUY",
                        "price": float(price), "weight_traded": new_w, "reason": "Rebal entry",
                    })
                else:
                    old_w = positions[ticker]["weight"]
                    delta = new_w - old_w
                    positions[ticker]["weight"] = new_w
                    if delta > 0:
                        buy_turnover += delta
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

        # Min exposure floor
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
                            "weight": fw, "entry_price": float(price),
                            "entry_date": current_date, "high_watermark": float(price),
                            "stop_price": stop_price, "losing_days": 0, "pyramid_count": 0,
                        }
                        buy_turnover += fw

        # Daily PnL
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
                    candidate_stop = positions[ticker]["high_watermark"] - config.atr_multiple * atr_val
                    if positions[ticker]["stop_price"] is None:
                        positions[ticker]["stop_price"] = float(candidate_stop)
                    else:
                        positions[ticker]["stop_price"] = max(positions[ticker]["stop_price"], float(candidate_stop))

                exit_reason = None
                if (config.use_atr_trailing_stop and positions[ticker]["stop_price"] is not None
                        and price < positions[ticker]["stop_price"]):
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
                        "date": current_date, "ticker": ticker, "action": "SELL",
                        "price": float(price * (1 - config.stop_exit_slippage)),
                        "weight_traded": positions[ticker]["weight"], "reason": exit_reason,
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
            "mom_bench_daily_return": mom_bench_returns.loc[current_date],
            "strategy_value": portfolio_value,
            "benchmark_value": benchmark_value,
            "mom_bench_value": mom_bench_value,
            "turnover": buy_turnover + sell_turnover,
            "positions_count": len(positions),
            "cash_weight": max(0.0, 1 - pos_weight_sum) if positions else 1.0,
            "portfolio_dd": portfolio_dd,
            "allocation_pct": allocation_pct,
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    trades_df = pd.DataFrame(trades_records)

    return {"daily": daily_df, "trades": trades_df}


def compute_metrics(daily_df: pd.DataFrame, rf: float = 0.04) -> dict:
    """Compute comprehensive metrics from daily equity curve."""
    curve = daily_df.set_index("date")["strategy_value"]
    bench = daily_df.set_index("date")["benchmark_value"]
    strat_ret = daily_df.set_index("date")["strategy_daily_return"]
    bench_ret = daily_df.set_index("date")["benchmark_daily_return"]

    cagr = annualized_return(curve)
    mdd = max_drawdown(curve)
    vol = annualized_vol(strat_ret)
    shp = sharpe_ratio(strat_ret, rf)
    daily_rf = (1 + rf) ** (1 / 252) - 1
    downside = strat_ret[strat_ret < 0].std(ddof=1) * np.sqrt(252)
    sortino = (strat_ret - daily_rf).mean() / strat_ret[strat_ret < 0].std(ddof=1) * np.sqrt(252) if downside > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    peak = curve.cummax()
    in_dd = curve < peak
    dd_groups = (~in_dd).cumsum()
    dd_dur = in_dd.groupby(dd_groups).sum()
    longest_dd = int(dd_dur.max()) if len(dd_dur) > 0 else 0

    avg_exposure = (1 - daily_df["cash_weight"]).mean()
    avg_positions = daily_df["positions_count"].mean()
    avg_turnover = daily_df["turnover"].mean()

    # Yearly returns
    daily_df_idx = daily_df.set_index("date")
    yearly = daily_df_idx["strategy_value"].resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()
    bench_yearly = daily_df_idx["benchmark_value"].resample("YE").last()
    bench_yearly_ret = bench_yearly.pct_change().dropna()

    yearly_data = []
    for dt in yearly_ret.index:
        yr = dt.year
        sr = yearly_ret.loc[dt]
        br = bench_yearly_ret.loc[dt] if dt in bench_yearly_ret.index else np.nan
        yearly_data.append({"year": yr, "strategy": sr, "benchmark": br, "alpha": sr - br})

    # Drawdown episodes
    dd_series = curve / peak - 1
    episodes = []
    in_episode = False
    ep_start = None
    for dt in dd_series.index:
        if dd_series[dt] < -0.05 and not in_episode:
            in_episode = True
            ep_start = dt
        elif dd_series[dt] >= 0 and in_episode:
            in_episode = False
            ep_end = dt
            trough_idx = dd_series[ep_start:ep_end].idxmin()
            trough_val = dd_series[trough_idx]
            duration = (ep_end - ep_start).days
            recovery = (ep_end - trough_idx).days
            episodes.append({
                "start": ep_start.strftime("%Y-%m-%d"),
                "trough": trough_idx.strftime("%Y-%m-%d"),
                "end": ep_end.strftime("%Y-%m-%d"),
                "depth": trough_val,
                "duration_days": duration,
                "recovery_days": recovery,
            })
    if in_episode:
        trough_idx = dd_series[ep_start:].idxmin()
        episodes.append({
            "start": ep_start.strftime("%Y-%m-%d"),
            "trough": trough_idx.strftime("%Y-%m-%d"),
            "end": "ongoing",
            "depth": dd_series[trough_idx],
            "duration_days": (dd_series.index[-1] - ep_start).days,
            "recovery_days": None,
        })

    # Monthly returns
    monthly_ret = (1 + strat_ret).resample("ME").prod() - 1
    bench_monthly = (1 + bench_ret).resample("ME").prod() - 1
    monthly_alpha = monthly_ret - bench_monthly
    monthly_hit = (monthly_alpha > 0).mean()

    # Rolling 3-year CAGR
    rolling_3yr = []
    curve_vals = curve.values
    curve_idx = curve.index
    for j in range(756, len(curve_vals)):
        v_start = curve_vals[j - 756]
        v_end = curve_vals[j]
        r3 = (v_end / v_start) ** (1 / 3) - 1
        rolling_3yr.append({"date": curve_idx[j], "rolling_3yr_cagr": r3})

    return {
        "cagr": cagr,
        "bench_cagr": annualized_return(bench),
        "max_dd": mdd,
        "bench_mdd": max_drawdown(bench),
        "longest_dd": longest_dd,
        "vol": vol,
        "sharpe": shp,
        "sortino": sortino,
        "calmar": calmar,
        "avg_exposure": avg_exposure,
        "avg_positions": avg_positions,
        "avg_turnover": avg_turnover,
        "final_value": curve.iloc[-1],
        "monthly_hit_rate": monthly_hit,
        "yearly_data": yearly_data,
        "drawdown_episodes": episodes,
        "rolling_3yr": rolling_3yr,
        "best_year": max(yearly_data, key=lambda x: x["strategy"])["strategy"] if yearly_data else np.nan,
        "worst_year": min(yearly_data, key=lambda x: x["strategy"])["strategy"] if yearly_data else np.nan,
        "positive_years": sum(1 for y in yearly_data if y["strategy"] > 0),
        "total_years": len(yearly_data),
    }


def main():
    t0 = time.time()
    config = load_config_from_yaml()
    config = replace(config, start_date="2006-04-01")
    logger.info("Config loaded. 20-year backtest: %s to %s", config.start_date, config.end_date)

    yf_universe = _yf_tickers(BROAD_UNIVERSE)
    extra = [config.benchmark]
    if config.momentum_benchmark:
        extra.append(config.momentum_benchmark)
    tickers = yf_universe + extra

    logger.info("Downloading data for %d tickers...", len(tickers))
    raw_data = download_ohlcv(tickers, config.start_date, config.end_date)
    logger.info("Download complete in %.0fs.", time.time() - t0)

    # Run baseline
    logger.info("Running BASELINE backtest...")
    t1 = time.time()
    base_result = run_backtest_core(config, raw_data, use_gradient_dd=False)
    base_metrics = compute_metrics(base_result["daily"])
    logger.info("  Baseline done in %.0fs. CAGR=%.2f%%", time.time() - t1, base_metrics["cagr"] * 100)

    # Run Mod 07
    logger.info("Running MOD 07 (Gradient DD) backtest...")
    t2 = time.time()
    mod7_result = run_backtest_core(config, raw_data, use_gradient_dd=True)
    mod7_metrics = compute_metrics(mod7_result["daily"])
    logger.info("  Mod 07 done in %.0fs. CAGR=%.2f%%", time.time() - t2, mod7_metrics["cagr"] * 100)

    # ── Print Report ──
    b = base_metrics
    m = mod7_metrics

    print("\n" + "=" * 100)
    print("20-YEAR BACKTEST REPORT: BASELINE vs MOD 07 (GRADIENT DRAWDOWN ALLOCATION)")
    print(f"Period: {config.start_date} to {config.end_date or 'present'}  |  Initial Capital: Rs {config.initial_capital:,.0f}")
    print("=" * 100)

    print("\n╔══════════════════════════════════╦═══════════════╦═══════════════╦══════════════╗")
    print("║ Metric                           ║    Baseline   ║    Mod 07     ║   Delta      ║")
    print("╠══════════════════════════════════╬═══════════════╬═══════════════╬══════════════╣")

    rows = [
        ("CAGR", f"{b['cagr']*100:.2f}%", f"{m['cagr']*100:.2f}%", f"{(m['cagr']-b['cagr'])*100:+.2f}pp"),
        ("Benchmark CAGR", f"{b['bench_cagr']*100:.2f}%", f"{m['bench_cagr']*100:.2f}%", "--"),
        ("Final Value", f"Rs {b['final_value']:,.0f}", f"Rs {m['final_value']:,.0f}", f"{(m['final_value']/b['final_value']-1)*100:+.1f}%"),
        ("Max Drawdown", f"{b['max_dd']*100:.2f}%", f"{m['max_dd']*100:.2f}%", f"{(m['max_dd']-b['max_dd'])*100:+.2f}pp"),
        ("Longest DD (days)", f"{b['longest_dd']}", f"{m['longest_dd']}", f"{m['longest_dd']-b['longest_dd']:+d}"),
        ("Annualized Volatility", f"{b['vol']*100:.2f}%", f"{m['vol']*100:.2f}%", f"{(m['vol']-b['vol'])*100:+.2f}pp"),
        ("Sharpe (rf=4%)", f"{b['sharpe']:.2f}", f"{m['sharpe']:.2f}", f"{m['sharpe']-b['sharpe']:+.2f}"),
        ("Sortino (rf=4%)", f"{b['sortino']:.2f}", f"{m['sortino']:.2f}", f"{m['sortino']-b['sortino']:+.2f}"),
        ("Calmar Ratio", f"{b['calmar']:.2f}", f"{m['calmar']:.2f}", f"{m['calmar']-b['calmar']:+.2f}"),
        ("Monthly Hit Rate vs Bench", f"{b['monthly_hit_rate']*100:.1f}%", f"{m['monthly_hit_rate']*100:.1f}%", f"{(m['monthly_hit_rate']-b['monthly_hit_rate'])*100:+.1f}pp"),
        ("Avg Exposure", f"{b['avg_exposure']*100:.1f}%", f"{m['avg_exposure']*100:.1f}%", f"{(m['avg_exposure']-b['avg_exposure'])*100:+.1f}pp"),
        ("Avg Positions", f"{b['avg_positions']:.1f}", f"{m['avg_positions']:.1f}", f"{m['avg_positions']-b['avg_positions']:+.1f}"),
        ("Avg Daily Turnover", f"{b['avg_turnover']*100:.2f}%", f"{m['avg_turnover']*100:.2f}%", f"{(m['avg_turnover']-b['avg_turnover'])*100:+.2f}pp"),
        ("Best Year", f"{b['best_year']*100:.1f}%", f"{m['best_year']*100:.1f}%", ""),
        ("Worst Year", f"{b['worst_year']*100:.1f}%", f"{m['worst_year']*100:.1f}%", ""),
        ("Positive Years", f"{b['positive_years']}/{b['total_years']}", f"{m['positive_years']}/{m['total_years']}", ""),
    ]

    for label, bv, mv, dv in rows:
        print(f"║ {label:<32} ║ {bv:>13} ║ {mv:>13} ║ {dv:>12} ║")

    print("╚══════════════════════════════════╩═══════════════╩═══════════════╩══════════════╝")

    # Year-by-year comparison
    print("\n\n" + "=" * 90)
    print("YEAR-BY-YEAR RETURNS")
    print("=" * 90)
    print(f"{'Year':>6} {'Baseline':>12} {'Mod 07':>12} {'Benchmark':>12} {'Base Alpha':>12} {'Mod7 Alpha':>12} {'Winner':>10}")
    print("-" * 90)

    base_yearly = {y["year"]: y for y in b["yearly_data"]}
    mod7_yearly = {y["year"]: y for y in m["yearly_data"]}
    all_years = sorted(set(list(base_yearly.keys()) + list(mod7_yearly.keys())))

    base_wins = 0
    mod7_wins = 0
    for yr in all_years:
        by = base_yearly.get(yr, {})
        my = mod7_yearly.get(yr, {})
        bs = by.get("strategy", np.nan)
        ms = my.get("strategy", np.nan)
        bn = by.get("benchmark", np.nan)
        ba = by.get("alpha", np.nan)
        ma = my.get("alpha", np.nan) if my else np.nan

        if pd.notna(bs) and pd.notna(ms):
            winner = "Mod 07" if ms > bs else "Baseline" if bs > ms else "Tie"
            if ms > bs:
                mod7_wins += 1
            elif bs > ms:
                base_wins += 1
        else:
            winner = "--"

        print(
            f"{yr:>6} "
            f"{bs*100:>11.2f}% " if pd.notna(bs) else f"{'N/A':>12} ",
            end=""
        )
        print(
            f"{ms*100:>11.2f}% " if pd.notna(ms) else f"{'N/A':>12} ",
            end=""
        )
        print(
            f"{bn*100:>11.2f}% " if pd.notna(bn) else f"{'N/A':>12} ",
            end=""
        )
        print(
            f"{ba*100:>11.2f}% " if pd.notna(ba) else f"{'N/A':>12} ",
            end=""
        )
        print(
            f"{ma*100:>11.2f}% " if pd.notna(ma) else f"{'N/A':>12} ",
            end=""
        )
        print(f"{winner:>10}")

    print("-" * 90)
    print(f"Year wins: Baseline={base_wins}, Mod 07={mod7_wins}")

    # Drawdown episodes comparison
    print("\n\n" + "=" * 100)
    print("MAJOR DRAWDOWN EPISODES (>5% depth)")
    print("=" * 100)

    print("\n--- BASELINE ---")
    print(f"{'Start':<12} {'Trough':<12} {'End':<12} {'Depth':>8} {'Duration':>10} {'Recovery':>10}")
    print("-" * 70)
    for ep in b["drawdown_episodes"]:
        rec = f"{ep['recovery_days']}d" if ep['recovery_days'] is not None else "ongoing"
        print(f"{ep['start']:<12} {ep['trough']:<12} {ep['end']:<12} {ep['depth']*100:>7.2f}% {ep['duration_days']:>9}d {rec:>10}")

    print(f"\nTotal major drawdown episodes: {len(b['drawdown_episodes'])}")

    print("\n--- MOD 07 (GRADIENT DD) ---")
    print(f"{'Start':<12} {'Trough':<12} {'End':<12} {'Depth':>8} {'Duration':>10} {'Recovery':>10}")
    print("-" * 70)
    for ep in m["drawdown_episodes"]:
        rec = f"{ep['recovery_days']}d" if ep['recovery_days'] is not None else "ongoing"
        print(f"{ep['start']:<12} {ep['trough']:<12} {ep['end']:<12} {ep['depth']*100:>7.2f}% {ep['duration_days']:>9}d {rec:>10}")

    print(f"\nTotal major drawdown episodes: {len(m['drawdown_episodes'])}")

    # Rolling 3-year CAGR stats
    print("\n\n" + "=" * 80)
    print("ROLLING 3-YEAR CAGR STATISTICS")
    print("=" * 80)

    if b["rolling_3yr"] and m["rolling_3yr"]:
        b_r3 = pd.DataFrame(b["rolling_3yr"])["rolling_3yr_cagr"]
        m_r3 = pd.DataFrame(m["rolling_3yr"])["rolling_3yr_cagr"]

        print(f"{'Stat':<25} {'Baseline':>15} {'Mod 07':>15}")
        print("-" * 55)
        print(f"{'Median 3yr CAGR':<25} {b_r3.median()*100:>14.2f}% {m_r3.median()*100:>14.2f}%")
        print(f"{'Mean 3yr CAGR':<25} {b_r3.mean()*100:>14.2f}% {m_r3.mean()*100:>14.2f}%")
        print(f"{'Min 3yr CAGR':<25} {b_r3.min()*100:>14.2f}% {m_r3.min()*100:>14.2f}%")
        print(f"{'Max 3yr CAGR':<25} {b_r3.max()*100:>14.2f}% {m_r3.max()*100:>14.2f}%")
        print(f"{'% periods > 15% CAGR':<25} {(b_r3>0.15).mean()*100:>14.1f}% {(m_r3>0.15).mean()*100:>14.1f}%")
        print(f"{'% periods > 20% CAGR':<25} {(b_r3>0.20).mean()*100:>14.1f}% {(m_r3>0.20).mean()*100:>14.1f}%")
        print(f"{'% periods negative':<25} {(b_r3<0).mean()*100:>14.1f}% {(m_r3<0).mean()*100:>14.1f}%")

    # Final summary
    print("\n\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    mult_b = b["final_value"] / config.initial_capital
    mult_m = m["final_value"] / config.initial_capital
    print(f"\nOver 20 years, Rs {config.initial_capital/1e7:.0f} Cr invested:")
    print(f"  Baseline:         grew to Rs {b['final_value']/1e7:.2f} Cr  ({mult_b:.1f}x)")
    print(f"  Mod 07 (Grad DD): grew to Rs {m['final_value']/1e7:.2f} Cr  ({mult_m:.1f}x)")
    print(f"  Benchmark:        grew to Rs {base_result['daily'].iloc[-1]['benchmark_value']/1e7:.2f} Cr")
    print(f"\nMod 07 produced {(m['final_value']/b['final_value']-1)*100:.1f}% more terminal wealth than Baseline.")
    print(f"Mod 07 won {mod7_wins}/{len(all_years)} calendar years vs Baseline's {base_wins}/{len(all_years)}.")

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
