"""Full momentum backtest driven by config.yaml.

Loads the Config dataclass from config.yaml so that backtests and
the live system share the exact same parameters.  Change config.yaml
once, both systems reflect it.

Usage:
    python main.py backtest          # via CLI
    python backtest.py               # standalone
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

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

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Strategy configuration loaded from config.yaml."""

    start_date: str = "2014-01-01"
    end_date: Optional[str] = None
    benchmark: str = "^CNX200"
    momentum_benchmark: str = "NIFTY200MOMENTM30.NS"

    initial_capital: float = 2_00_00_000
    annual_addition: float = 0.0
    addition_month: int = 4

    top_liquid_n: int = 120
    top_momentum_n: int = 12
    breadth_threshold: float = 0.40
    neutral_breadth_threshold: float = 0.30
    neutral_allocation_pct: float = 0.50
    require_positive_3m_return: bool = True

    annual_cash_yield: float = 0.04
    buy_side_cost: float = 0.001554
    sell_side_cost: float = 0.002354
    max_weight: float = 0.15
    min_weight: float = 0.04
    score_blend: float = 0.4
    max_sector_weight: float = 0.30

    use_50dma_exit: bool = False
    use_atr_trailing_stop: bool = True
    atr_window: int = 14
    atr_multiple: float = 2.0
    max_loss_pct: float = -0.08
    max_hold_losing_days: int = 10
    stop_exit_slippage: float = 0.005

    min_exposure: float = 0.30
    min_exposure_slots: int = 3

    drawdown_circuit_breaker: float = -0.20
    cb_reset_days: int = 5
    re_entry_enabled: bool = True

    momentum_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    continuity_bonus: float = 0.3
    sector_downtrend_penalty: float = 0.5
    max_vol_percentile: float = 0.90

    pyramid_enabled: bool = True
    pyramid_threshold_pct: float = 0.05
    pyramid_add_pct: float = 0.03
    pyramid_max: int = 2
    pyramid_ratchet_stop: bool = True

    rebal_offset_from_end: int = 1  # 1 = last trading day, 3 = 3rd-to-last

    ffill_limit: int = 5
    min_price: float = 100.0
    output_file: str = "dynamic_momentum_daily_exit_backtest.xlsx"


def load_config_from_yaml(yaml_path: str | Path | None = None) -> Config:
    """Load a Config from config.yaml, mapping YAML sections to the flat
    Config dataclass fields.
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "config.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    strat = raw.get("strategy", {})
    regime = raw.get("regime", {})
    sizing = raw.get("sizing", {})
    exits = raw.get("exits", {})
    risk = raw.get("risk", {})
    costs = raw.get("costs", {})
    capital = raw.get("capital", {})
    pyramid = raw.get("pyramid", {})

    brokerage = costs.get("one_way_brokerage", 0.0003)
    slippage = costs.get("slippage_estimate", 0.001)
    gst_rate = costs.get("gst_rate", 0.18)
    stt_sell = costs.get("stt_sell_side", 0.001)
    stamp_buy = costs.get("stamp_duty_buy", 0.00015)

    brokerage_with_gst = brokerage * (1 + gst_rate)
    buy_cost = brokerage_with_gst + stamp_buy + slippage
    sell_cost = brokerage_with_gst + stt_sell + slippage

    return Config(
        benchmark="^CNX200",
        momentum_benchmark="NIFTY200MOMENTM30.NS",
        initial_capital=capital.get("initial", 2_00_00_000),
        annual_addition=capital.get("annual_addition", 0),
        addition_month=capital.get("addition_month", 4),
        top_liquid_n=strat.get("top_liquid_n", 120),
        top_momentum_n=strat.get("top_momentum_n", 12),
        breadth_threshold=regime.get("breadth_threshold", 0.40),
        neutral_breadth_threshold=regime.get("neutral_breadth_threshold", 0.30),
        neutral_allocation_pct=regime.get("neutral_allocation_pct", 0.50),
        require_positive_3m_return=regime.get("require_positive_3m_return", True),
        annual_cash_yield=capital.get("cash_yield", 0.04),
        buy_side_cost=buy_cost,
        sell_side_cost=sell_cost,
        max_weight=sizing.get("max_weight_per_stock", 0.15),
        min_weight=sizing.get("min_weight_per_stock", 0.04),
        score_blend=sizing.get("score_blend", 0.4),
        max_sector_weight=sizing.get("max_sector_weight", 0.30),
        use_50dma_exit=exits.get("use_50dma_exit", False),
        use_atr_trailing_stop=exits.get("use_atr_trailing_stop", True),
        atr_window=exits.get("atr_window", 14),
        atr_multiple=exits.get("atr_multiple", 2.0),
        max_loss_pct=exits.get("max_loss_pct", -0.08),
        max_hold_losing_days=exits.get("max_hold_losing_days", 10),
        stop_exit_slippage=exits.get("stop_exit_slippage", 0.005),
        min_exposure=risk.get("min_exposure", 0.30),
        min_exposure_slots=risk.get("min_exposure_slots", 3),
        drawdown_circuit_breaker=risk.get("drawdown_circuit_breaker", -0.20),
        cb_reset_days=risk.get("cb_reset_days", 5),
        re_entry_enabled=risk.get("re_entry_enabled", True),
        momentum_weights=strat.get("momentum_weights", [0.4, 0.3, 0.2, 0.1]),
        continuity_bonus=strat.get("continuity_bonus", 0.3),
        sector_downtrend_penalty=strat.get("sector_downtrend_penalty", 0.5),
        max_vol_percentile=strat.get("max_vol_percentile", 0.90),
        pyramid_enabled=pyramid.get("enabled", True),
        pyramid_threshold_pct=pyramid.get("threshold_pct", 0.05),
        pyramid_add_pct=pyramid.get("add_pct", 0.03),
        pyramid_max=pyramid.get("max_pyramids", 2),
        pyramid_ratchet_stop=pyramid.get("ratchet_stop_to_breakeven", True),
        min_price=strat.get("min_price", 100.0),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _yf_tickers(symbols: list[str]) -> list[str]:
    """Convert plain NSE symbols to yfinance format (append .NS)."""
    return [
        f"{s}.NS" if not s.endswith(".NS") and not s.startswith("^") else s
        for s in symbols
    ]


def _yf_sector_map(sector_map: dict[str, str]) -> dict[str, str]:
    return {f"{k}.NS": v for k, v in sector_map.items()}


def download_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance.

    Returns a dict keyed by field name (Open, High, Low, Close,
    Adj Close, Volume) with DataFrames of shape (dates x tickers).
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    if data.empty:
        raise ValueError(
            "yfinance returned an empty DataFrame — check tickers and date range"
        )

    is_multi = isinstance(data.columns, pd.MultiIndex)
    is_single = len(tickers) == 1

    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out: dict[str, pd.DataFrame] = {}
    for fld in fields:
        df = pd.DataFrame(index=data.index)
        for t in tickers:
            if is_multi:
                key = (t, fld)
                if key in data.columns:
                    df[t] = data[key]
                else:
                    logger.warning("Missing column %s for ticker %s", fld, t)
                    df[t] = np.nan
            elif is_single:
                df[t] = data[fld] if fld in data.columns else np.nan
            else:
                df[t] = np.nan
        out[fld] = df.sort_index()
    return out


# ---------------------------------------------------------------------------
# Full momentum backtest
# ---------------------------------------------------------------------------

def run_backtest(config: Config) -> dict:
    """Run the full momentum backtest using the given Config and return
    a dict of result DataFrames suitable for Excel export.
    """
    yf_universe = _yf_tickers(BROAD_UNIVERSE)
    sector_map = _yf_sector_map(SECTOR_MAP)

    extra_tickers = [config.benchmark]
    if config.momentum_benchmark:
        extra_tickers.append(config.momentum_benchmark)
    tickers = yf_universe + extra_tickers

    logger.info("Downloading data for %d tickers …", len(tickers))
    raw = download_ohlcv(tickers, config.start_date, config.end_date)

    close = raw["Adj Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    volume = raw["Volume"].copy()

    if config.benchmark not in close.columns:
        raise ValueError(
            f"Benchmark '{config.benchmark}' not found in downloaded data. "
            f"Available columns: {list(close.columns)}"
        )

    has_mom_bench = (
        config.momentum_benchmark
        and config.momentum_benchmark in close.columns
        and close[config.momentum_benchmark].dropna().shape[0] > 252
    )
    if config.momentum_benchmark and not has_mom_bench:
        logger.warning(
            "Momentum benchmark '%s' not available or insufficient data — skipping",
            config.momentum_benchmark,
        )

    benchmark = close[config.benchmark].dropna()
    cols_to_drop = [config.benchmark]
    if config.momentum_benchmark and config.momentum_benchmark in close.columns:
        cols_to_drop.append(config.momentum_benchmark)

    stocks_close_raw = close.drop(columns=cols_to_drop).reindex(benchmark.index)
    stocks_close = stocks_close_raw.ffill(limit=config.ffill_limit)

    _missing_mask = stocks_close_raw.isna()
    _missing_streak = (
        _missing_mask
        .apply(lambda col: col.groupby((~col).cumsum()).cumsum())
        .fillna(0)
        .astype(int)
    )

    stocks_high = high.drop(columns=[config.benchmark]).reindex(benchmark.index)
    stocks_low = low.drop(columns=[config.benchmark]).reindex(benchmark.index)
    stocks_volume = (
        volume.drop(columns=[config.benchmark]).reindex(benchmark.index).fillna(0)
    )

    daily_returns = stocks_close.pct_change().fillna(0)
    benchmark_returns = benchmark.pct_change().fillna(0)

    if has_mom_bench:
        mom_bench_series = (
            close[config.momentum_benchmark]
            .dropna()
            .reindex(benchmark.index)
            .ffill()
        )
        mom_bench_returns = mom_bench_series.pct_change().fillna(0)
    else:
        mom_bench_returns = pd.Series(0.0, index=benchmark.index)

    # Pre-compute indicators
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

    rebalance_records: list[dict] = []
    picks_records: list[dict] = []
    trades_records: list[dict] = []
    daily_records: list[dict] = []
    regime_records: list[dict] = []

    mw = config.momentum_weights
    start_index = max(252, 200)

    for i in range(start_index, len(benchmark.index)):
        current_date = benchmark.index[i]
        buy_turnover = 0.0
        sell_turnover = 0.0

        # Yearly capital addition
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

        # ── Drawdown circuit breaker ──
        portfolio_peak = max(portfolio_peak, portfolio_value)
        portfolio_dd = (
            (portfolio_value / portfolio_peak) - 1 if portfolio_peak > 0 else 0.0
        )

        if circuit_breaker_active:
            if full_risk_on:
                cb_risk_on_streak += 1
            else:
                cb_risk_on_streak = 0
            if cb_risk_on_streak >= config.cb_reset_days:
                circuit_breaker_active = False
                cb_risk_on_streak = 0
                portfolio_peak = portfolio_value
                logger.info(
                    "Circuit breaker RESET on %s (%d consecutive risk-on days, peak reset to %.0f)",
                    current_date.date(),
                    config.cb_reset_days,
                    portfolio_value,
                )
            else:
                allocation_pct = 0.0
        elif portfolio_dd < config.drawdown_circuit_breaker:
            circuit_breaker_active = True
            allocation_pct = 0.0
            logger.warning(
                "Circuit breaker TRIGGERED on %s (DD: %.1f%%)",
                current_date.date(),
                portfolio_dd * 100,
            )
            for ticker in list(positions.keys()):
                trades_records.append({
                    "date": current_date,
                    "ticker": ticker,
                    "action": "SELL",
                    "price": stocks_close.loc[current_date, ticker],
                    "weight_traded": positions[ticker]["weight"],
                    "reason": "Circuit breaker liquidation",
                })
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]

        regime_records.append({
            "date": current_date,
            "risk_on": full_risk_on,
            "allocation_pct": allocation_pct,
            "breadth": breadth,
            "benchmark_close": benchmark.loc[current_date],
            "benchmark_50dma": bench_50.loc[current_date],
            "benchmark_200dma": bench_200.loc[current_date],
            "benchmark_3m_return": bench_63ret.loc[current_date],
        })

        # ── Process pending exits (T+1 execution) ──
        for ticker in pending_exits:
            if ticker in positions:
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]
        pending_exits = []

        # ── Force-exit stocks with extended missing data ──
        for ticker in list(positions.keys()):
            if _missing_streak.loc[current_date, ticker] > config.ffill_limit:
                logger.warning(
                    "Force-exit %s on %s: %d consecutive missing days",
                    ticker,
                    current_date.date(),
                    _missing_streak.loc[current_date, ticker],
                )
                trades_records.append({
                    "date": current_date,
                    "ticker": ticker,
                    "action": "SELL",
                    "price": stocks_close.loc[current_date, ticker],
                    "weight_traded": positions[ticker]["weight"],
                    "reason": "Extended missing data",
                })
                sell_turnover += positions[ticker]["weight"]
                del positions[ticker]

        # Reset stopped-out tracker on new calendar month
        cur_ym = (current_date.year, current_date.month)
        if cur_ym != last_rebal_month and last_rebal_month != (0, 0):
            stopped_out_this_month = {}

        # ── Monthly rebalance ──
        if current_date in rebalance_dates:
            stopped_out_this_month = {}
            last_rebal_month = cur_ym

            liquid = adv_126.loc[current_date].dropna().sort_values(ascending=False)
            liquid = liquid[
                stocks_close.loc[current_date, liquid.index] >= config.min_price
            ]
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

                vol_20 = stocks_volume.iloc[max(0, i - 20) : i + 1].mean()
                vol_60_avg = stocks_volume.iloc[max(0, i - 60) : i + 1].mean()
                rel_volume = (
                    (vol_20 / vol_60_avg.replace(0, np.nan)).reindex(lu)
                )
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

                if config.sector_downtrend_penalty < 1.0:
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
                max_per_sector = max(
                    1, int(num_slots * config.max_sector_weight) + 1
                )
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
                else:
                    new_weights = pd.Series(dtype=float)
            else:
                liquid_universe = []
                selected = []
                new_weights = pd.Series(dtype=float)

            old_weights_snap = {t: positions[t]["weight"] for t in positions}
            existing = set(positions.keys())
            target = set(selected)

            for ticker in list(existing - target):
                exit_w = old_weights_snap.get(ticker, 0.0)
                trades_records.append({
                    "date": current_date,
                    "ticker": ticker,
                    "action": "REBAL_EXIT",
                    "price": stocks_close.loc[current_date, ticker],
                    "weight_traded": exit_w,
                    "reason": "Dropped from target universe",
                })
                sell_turnover += exit_w
                del positions[ticker]

            for ticker in selected:
                price = stocks_close.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]
                new_w = float(new_weights.loc[ticker])
                old_w = old_weights_snap.get(ticker, 0.0)

                if ticker not in positions:
                    stop_price = None
                    if config.use_atr_trailing_stop and pd.notna(atr_val):
                        stop_price = float(price - config.atr_multiple * atr_val)

                    positions[ticker] = {
                        "weight": new_w,
                        "entry_price": float(price),
                        "high_watermark": float(price),
                        "stop_price": stop_price,
                        "losing_days": 0,
                        "pyramid_count": 0,
                    }
                    trades_records.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "price": price,
                        "weight_traded": new_w,
                        "reason": "Monthly rebalance entry",
                    })
                    buy_turnover += new_w
                else:
                    delta = new_w - old_w
                    positions[ticker]["weight"] = new_w
                    if delta > 0:
                        buy_turnover += delta
                    else:
                        sell_turnover += abs(delta)

                picks_records.append({
                    "rebalance_date": current_date,
                    "ticker": ticker,
                    "weight": new_w,
                    "score": float(cand.loc[ticker, "risk_adj_score"]) if ticker in cand.index else np.nan,
                    "raw_momentum": float(cand.loc[ticker, "score"]) if ticker in cand.index else np.nan,
                    "volatility_60d": float(cand.loc[ticker, "vol"]) if ticker in cand.index else np.nan,
                    "adv_126": float(adv_126.loc[current_date, ticker]) if ticker in adv_126.columns else np.nan,
                })

            rebalance_records.append({
                "rebalance_date": current_date,
                "risk_on": full_risk_on,
                "allocation_pct": allocation_pct,
                "breadth": breadth,
                "liquid_universe_size": len(liquid_universe),
                "num_selected": len(selected),
            })

        # ── Daily re-entry for stopped-out stocks ──
        if (
            config.re_entry_enabled
            and stopped_out_this_month
            and not last_rebal_cand.empty
        ):
            top_ranked = set(
                last_rebal_cand.head(config.top_momentum_n).index
            )
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
                        stop_price = float(
                            price - config.atr_multiple * atr_val
                        )
                    positions[ticker] = {
                        "weight": orig_w,
                        "entry_price": float(price),
                        "high_watermark": float(price),
                        "stop_price": stop_price,
                        "losing_days": 0,
                        "pyramid_count": 0,
                    }
                    trades_records.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "price": price,
                        "weight_traded": orig_w,
                        "reason": "Re-entry after ATR stop",
                    })
                    buy_turnover += orig_w

        # ── Mid-month pyramiding ──
        if (
            config.pyramid_enabled
            and not last_rebal_cand.empty
            and allocation_pct > 0
            and current_date not in rebalance_dates
        ):
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

                    trades_records.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "price": price,
                        "weight_traded": config.pyramid_add_pct,
                        "reason": f"Pyramid #{pos['pyramid_count']} (gain {gain:.1%})",
                    })

        # ── Minimum exposure floor ──
        if not last_rebal_cand.empty and allocation_pct > 0:
            pos_exposure = sum(p["weight"] for p in positions.values())
            if pos_exposure < config.min_exposure:
                needed = config.min_exposure - pos_exposure
                already_held = set(positions.keys()) | set(pending_exits)
                floor_candidates = [
                    t
                    for t in last_rebal_cand.index
                    if t not in already_held
                    and pd.notna(stocks_close.loc[current_date, t])
                    and stocks_close.loc[current_date, t] >= config.min_price
                ][: config.min_exposure_slots]
                if floor_candidates:
                    floor_vols = last_rebal_cand.loc[floor_candidates, "vol"]
                    floor_weights = capped_inverse_vol_weights(
                        floor_vols, config.max_weight
                    )
                    floor_weights = floor_weights / floor_weights.sum() * needed
                    for ticker in floor_candidates:
                        fw = float(floor_weights.loc[ticker])
                        price = stocks_close.loc[current_date, ticker]
                        atr_val = atr.loc[current_date, ticker]
                        stop_price = None
                        if config.use_atr_trailing_stop and pd.notna(atr_val):
                            stop_price = float(
                                price - config.atr_multiple * atr_val
                            )
                        positions[ticker] = {
                            "weight": fw,
                            "entry_price": float(price),
                            "high_watermark": float(price),
                            "stop_price": stop_price,
                            "losing_days": 0,
                            "pyramid_count": 0,
                        }
                        trades_records.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "BUY",
                            "price": price,
                            "weight_traded": fw,
                            "reason": "Min exposure floor",
                        })
                        buy_turnover += fw

        # ── Daily exit signals + PnL ──
        gross_ret = 0.0

        if len(positions) == 0:
            gross_ret = daily_cash_ret
        else:
            for ticker in list(positions.keys()):
                price = stocks_close.loc[current_date, ticker]
                ma50 = dma_50.loc[current_date, ticker]
                atr_val = atr.loc[current_date, ticker]

                if pd.isna(price):
                    continue

                positions[ticker]["high_watermark"] = max(
                    positions[ticker]["high_watermark"], float(price)
                )

                if config.use_atr_trailing_stop and pd.notna(atr_val):
                    candidate_stop = (
                        positions[ticker]["high_watermark"]
                        - config.atr_multiple * atr_val
                    )
                    if positions[ticker]["stop_price"] is None:
                        positions[ticker]["stop_price"] = float(candidate_stop)
                    else:
                        positions[ticker]["stop_price"] = max(
                            positions[ticker]["stop_price"],
                            float(candidate_stop),
                        )

                exit_reason = None
                if config.use_50dma_exit and pd.notna(ma50) and price < ma50:
                    exit_reason = "Price < 50DMA"
                elif (
                    config.use_atr_trailing_stop
                    and positions[ticker]["stop_price"] is not None
                    and price < positions[ticker]["stop_price"]
                ):
                    exit_reason = "ATR Trailing Stop"

                if exit_reason is None and config.max_loss_pct is not None:
                    pnl_pct = (price / positions[ticker]["entry_price"]) - 1
                    if pnl_pct < config.max_loss_pct:
                        exit_reason = f"Hard stop ({config.max_loss_pct:.0%})"

                if price < positions[ticker]["entry_price"]:
                    positions[ticker]["losing_days"] += 1
                else:
                    positions[ticker]["losing_days"] = 0

                if config.max_hold_losing_days > 0 and exit_reason is None and positions[ticker]["losing_days"] >= config.max_hold_losing_days:
                    exit_reason = f"Time stop ({config.max_hold_losing_days}d underwater)"

                if exit_reason is not None:
                    stopped_out_this_month[ticker] = positions[ticker]["weight"]
                    pending_exits.append(ticker)
                    exit_price = float(price * (1 - config.stop_exit_slippage))
                    trades_records.append({
                        "date": current_date,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_price,
                        "weight_traded": positions[ticker]["weight"],
                        "reason": exit_reason,
                    })

            # Mark-to-market PnL
            pre_move_total = 0.0
            for ticker in positions:
                if "current_value" not in positions[ticker]:
                    positions[ticker]["current_value"] = (
                        portfolio_value * positions[ticker]["weight"]
                    )
                pre_move_total += positions[ticker]["current_value"]

            invested_frac = (
                pre_move_total / portfolio_value if portfolio_value > 0 else 0.0
            )

            for ticker in positions:
                dr_val = daily_returns.loc[current_date, ticker]
                w_frac = (
                    positions[ticker]["current_value"] / portfolio_value
                    if portfolio_value > 0
                    else 0.0
                )
                gross_ret += w_frac * dr_val
                positions[ticker]["current_value"] *= 1 + dr_val

            if invested_frac < 1:
                gross_ret += (1 - invested_frac) * daily_cash_ret

            # Recompute weights from mark-to-market values
            post_move_total = sum(
                positions[t]["current_value"] for t in positions
            )
            if post_move_total > 0:
                for ticker in positions:
                    positions[ticker]["weight"] = (
                        positions[ticker]["current_value"] / post_move_total
                    )

        trading_cost = (
            buy_turnover * config.buy_side_cost
            + sell_turnover * config.sell_side_cost
        )
        net_ret = gross_ret - trading_cost
        portfolio_value *= 1 + net_ret

        if positions:
            total_w = sum(positions[t]["weight"] for t in positions)
            for t in positions:
                positions[t]["current_value"] = portfolio_value * (
                    positions[t]["weight"] / total_w if total_w > 0 else 0
                )

        pos_weight_sum = (
            sum(positions[t]["weight"] for t in positions) if positions else 0.0
        )
        daily_records.append({
            "date": current_date,
            "strategy_daily_return": net_ret,
            "benchmark_daily_return": benchmark_returns.loc[current_date],
            "mom_bench_daily_return": mom_bench_returns.loc[current_date],
            "strategy_value": portfolio_value,
            "benchmark_value": benchmark_value,
            "mom_bench_value": mom_bench_value,
            "gross_return": gross_ret,
            "trading_cost": trading_cost,
            "turnover": buy_turnover + sell_turnover,
            "positions_count": len(positions),
            "cash_weight": max(0.0, 1 - pos_weight_sum) if positions else 1.0,
        })

    # ── Build result DataFrames ──
    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["exposure"] = 1 - daily_df["cash_weight"]

    trades_df = pd.DataFrame(trades_records)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])

    picks_df = pd.DataFrame(picks_records)
    if not picks_df.empty:
        picks_df["rebalance_date"] = pd.to_datetime(picks_df["rebalance_date"])

    rebal_df = pd.DataFrame(rebalance_records)
    if not rebal_df.empty:
        rebal_df["rebalance_date"] = pd.to_datetime(rebal_df["rebalance_date"])

    regime_df = pd.DataFrame(regime_records)
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    # Round-trip trades (weighted-average cost basis lifecycle tracker)
    roundtrip_rows: list[dict] = []
    if not trades_df.empty:
        open_rt: dict[str, dict] = {}
        for _, row in trades_df.sort_values("date").iterrows():
            ticker = row["ticker"]
            action = row["action"]
            price = row["price"]
            weight = row["weight_traded"]
            date = row["date"]
            if action == "BUY":
                if ticker not in open_rt:
                    open_rt[ticker] = {
                        "total_cost": price * weight,
                        "total_weight": weight,
                        "entry_date": date,
                        "pyramids": 0,
                    }
                else:
                    pos = open_rt[ticker]
                    pos["total_cost"] += price * weight
                    pos["total_weight"] += weight
                    pos["pyramids"] += 1
            elif action in ("SELL", "REBAL_EXIT"):
                if ticker in open_rt:
                    pos = open_rt.pop(ticker)
                    wavg_entry = pos["total_cost"] / pos["total_weight"]
                    ret = price / wavg_entry - 1 if wavg_entry > 0 else np.nan
                    hold = (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days
                    roundtrip_rows.append({
                        "ticker": ticker,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": wavg_entry,
                        "exit_price": price,
                        "weight": pos["total_weight"],
                        "return": ret,
                        "holding_days": hold,
                        "pyramids": pos["pyramids"],
                        "exit_reason": row.get("reason", action),
                    })
    roundtrip_df = pd.DataFrame(roundtrip_rows)
    if not roundtrip_df.empty:
        roundtrip_df["entry_date"] = pd.to_datetime(roundtrip_df["entry_date"])
        roundtrip_df["exit_date"] = pd.to_datetime(roundtrip_df["exit_date"])

    # ── Performance metrics ──
    strategy_curve = daily_df.set_index("date")["strategy_value"]
    benchmark_curve = daily_df.set_index("date")["benchmark_value"]
    strategy_daily = daily_df.set_index("date")["strategy_daily_return"]
    benchmark_daily = daily_df.set_index("date")["benchmark_daily_return"]

    value_cols = ["strategy_value", "benchmark_value"]
    ret_cols = ["strategy_daily_return", "benchmark_daily_return"]
    if has_mom_bench:
        value_cols.append("mom_bench_value")
        ret_cols.append("mom_bench_daily_return")

    monthly_comp = daily_df.set_index("date")[value_cols].resample("ME").last()
    monthly_returns = daily_df.set_index("date")[ret_cols]
    monthly_returns = (1 + monthly_returns).resample("ME").prod() - 1
    monthly_out = (
        monthly_comp.join(monthly_returns, how="left")
        .reset_index()
        .rename(columns={"date": "month"})
    )

    strat_cagr = annualized_return(strategy_curve)
    bench_cagr = annualized_return(benchmark_curve)
    strat_mdd = max_drawdown(strategy_curve)
    bench_mdd = max_drawdown(benchmark_curve)

    strat_calmar = strat_cagr / abs(strat_mdd) if strat_mdd != 0 else np.nan
    bench_calmar = bench_cagr / abs(bench_mdd) if bench_mdd != 0 else np.nan

    daily_rf = (1 + config.annual_cash_yield) ** (1 / 252) - 1

    strat_excess = strategy_daily - daily_rf
    strat_downside = strategy_daily[strategy_daily < 0].std(ddof=1) * np.sqrt(252)
    strat_sortino = (
        strat_excess.mean()
        / strategy_daily[strategy_daily < 0].std(ddof=1)
        * np.sqrt(252)
        if strat_downside > 0
        else np.nan
    )

    bench_excess = benchmark_daily - daily_rf
    bench_downside = benchmark_daily[benchmark_daily < 0].std(ddof=1) * np.sqrt(252)
    bench_sortino = (
        bench_excess.mean()
        / benchmark_daily[benchmark_daily < 0].std(ddof=1)
        * np.sqrt(252)
        if bench_downside > 0
        else np.nan
    )

    if has_mom_bench:
        mom_bench_daily = daily_df.set_index("date")["mom_bench_daily_return"]
        mom_bench_curve = daily_df.set_index("date")["mom_bench_value"]
        mb_cagr = annualized_return(mom_bench_curve)
        mb_mdd = max_drawdown(mom_bench_curve)
        mb_calmar = mb_cagr / abs(mb_mdd) if mb_mdd != 0 else np.nan
        mb_vol = annualized_vol(mom_bench_daily)
        mb_downside = mom_bench_daily[mom_bench_daily < 0].std(ddof=1) * np.sqrt(252)
        mb_sharpe = sharpe_ratio(mom_bench_daily, config.annual_cash_yield)
        mb_sortino = (
            (mom_bench_daily - daily_rf).mean()
            / mom_bench_daily[mom_bench_daily < 0].std(ddof=1)
            * np.sqrt(252)
            if mb_downside > 0
            else np.nan
        )
    else:
        mb_cagr = mb_mdd = mb_calmar = mb_vol = mb_downside = np.nan
        mb_sharpe = mb_sortino = np.nan

    strat_peak = strategy_curve.cummax()
    strat_in_dd = strategy_curve < strat_peak
    dd_groups = (~strat_in_dd).cumsum()
    dd_durations = strat_in_dd.groupby(dd_groups).sum()
    longest_dd_days = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    if not roundtrip_df.empty:
        trade_returns = roundtrip_df["return"].dropna()
        win_rate = (trade_returns > 0).mean() if len(trade_returns) else np.nan
        avg_winner = (
            trade_returns[trade_returns > 0].mean()
            if (trade_returns > 0).any()
            else 0.0
        )
        avg_loser = (
            trade_returns[trade_returns <= 0].mean()
            if (trade_returns <= 0).any()
            else 0.0
        )
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns <= 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_holding_period = roundtrip_df["holding_days"].mean()
    else:
        trade_returns = pd.Series(dtype=float)
        win_rate = avg_winner = avg_loser = np.nan
        profit_factor = np.nan
        avg_holding_period = np.nan

    monthly_alpha = (
        monthly_returns["strategy_daily_return"]
        - monthly_returns["benchmark_daily_return"]
    )
    monthly_hit_rate = (monthly_alpha > 0).mean()

    avg_turnover = (
        daily_df["turnover"].mean() if "turnover" in daily_df.columns else np.nan
    )

    exposure_series = 1 - daily_df["cash_weight"]
    avg_exposure = exposure_series.mean()
    pct_time_in_market = (exposure_series > 0).mean()

    total_trades = len(trades_df) if not trades_df.empty else 0

    if has_mom_bench:
        mom_monthly_alpha = (
            monthly_returns["strategy_daily_return"]
            - monthly_returns["mom_bench_daily_return"]
        )
        monthly_hit_rate_vs_mom = (mom_monthly_alpha > 0).mean()
    else:
        monthly_hit_rate_vs_mom = np.nan

    mb_label = "Mom Bench (N200M30)"
    summary = pd.DataFrame([
        {"Metric": "Start Date", "Strategy": str(strategy_curve.index.min().date()), "Benchmark": str(benchmark_curve.index.min().date()), mb_label: str(strategy_curve.index.min().date())},
        {"Metric": "End Date", "Strategy": str(strategy_curve.index.max().date()), "Benchmark": str(benchmark_curve.index.max().date()), mb_label: str(strategy_curve.index.max().date())},
        {"Metric": "Initial Capital", "Strategy": config.initial_capital, "Benchmark": config.initial_capital, mb_label: config.initial_capital},
        {"Metric": "Annual Addition", "Strategy": config.annual_addition, "Benchmark": config.annual_addition, mb_label: config.annual_addition},
        {"Metric": "Final Value", "Strategy": strategy_curve.iloc[-1], "Benchmark": benchmark_curve.iloc[-1], mb_label: mom_bench_curve.iloc[-1] if has_mom_bench else np.nan},
        {"Metric": "CAGR", "Strategy": strat_cagr, "Benchmark": bench_cagr, mb_label: mb_cagr},
        {"Metric": "Max Drawdown", "Strategy": strat_mdd, "Benchmark": bench_mdd, mb_label: mb_mdd},
        {"Metric": "Longest Drawdown (days)", "Strategy": longest_dd_days, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Annualized Volatility", "Strategy": annualized_vol(strategy_daily), "Benchmark": annualized_vol(benchmark_daily), mb_label: mb_vol},
        {"Metric": "Downside Volatility", "Strategy": strat_downside, "Benchmark": bench_downside, mb_label: mb_downside},
        {"Metric": "Sharpe (rf=4%)", "Strategy": sharpe_ratio(strategy_daily, config.annual_cash_yield), "Benchmark": sharpe_ratio(benchmark_daily, config.annual_cash_yield), mb_label: mb_sharpe},
        {"Metric": "Sortino (rf=4%)", "Strategy": strat_sortino, "Benchmark": bench_sortino, mb_label: mb_sortino},
        {"Metric": "Calmar Ratio", "Strategy": strat_calmar, "Benchmark": bench_calmar, mb_label: mb_calmar},
        {"Metric": "Win Rate (by trade)", "Strategy": win_rate, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Avg Winner", "Strategy": avg_winner, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Avg Loser", "Strategy": avg_loser, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Profit Factor", "Strategy": profit_factor, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Total Roundtrips", "Strategy": len(roundtrip_df), "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Monthly Hit Rate vs Nifty200", "Strategy": monthly_hit_rate, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Monthly Hit Rate vs MomBench", "Strategy": monthly_hit_rate_vs_mom, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Avg Daily Turnover", "Strategy": avg_turnover, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Average Daily Positions", "Strategy": daily_df["positions_count"].mean(), "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Average Exposure", "Strategy": avg_exposure, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "% Time in Market", "Strategy": pct_time_in_market, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Total Trades", "Strategy": total_trades, "Benchmark": np.nan, mb_label: np.nan},
        {"Metric": "Avg Holding Period (days)", "Strategy": avg_holding_period, "Benchmark": np.nan, mb_label: np.nan},
    ])

    return {
        "summary": summary,
        "daily_results": daily_df,
        "monthly_results": monthly_out,
        "monthly_picks": picks_df,
        "trades": trades_df,
        "roundtrip_trades": roundtrip_df,
        "rebalances": rebal_df,
        "regime": regime_df,
        "daily_prices": stocks_close.reset_index().rename(columns={"index": "date"}),
        "daily_volume": stocks_volume.reset_index().rename(columns={"index": "date"}),
    }


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(bt: dict, output_file: str) -> None:
    """Write all backtest result sheets to a formatted Excel file."""
    has_mom = "mom_bench_value" in bt["monthly_results"].columns

    with pd.ExcelWriter(
        output_file, engine="xlsxwriter", datetime_format="yyyy-mm-dd"
    ) as writer:
        bt["summary"].to_excel(writer, sheet_name="Summary", index=False)
        bt["daily_results"].to_excel(writer, sheet_name="Daily_Results", index=False)
        bt["monthly_results"].to_excel(writer, sheet_name="Monthly_Results", index=False)
        bt["monthly_picks"].to_excel(writer, sheet_name="Monthly_Picks", index=False)
        bt["trades"].to_excel(writer, sheet_name="Trades", index=False)
        if not bt["roundtrip_trades"].empty:
            bt["roundtrip_trades"].to_excel(writer, sheet_name="Roundtrip_Trades", index=False)
        bt["rebalances"].to_excel(writer, sheet_name="Rebalances", index=False)
        bt["regime"].to_excel(writer, sheet_name="Regime", index=False)
        bt["daily_prices"].to_excel(writer, sheet_name="Daily_Prices", index=False)
        bt["daily_volume"].to_excel(writer, sheet_name="Daily_Volume", index=False)

        workbook = writer.book
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        money_fmt = workbook.add_format({"num_format": "#,##0.00"})
        date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})

        writer.sheets["Summary"].set_column("A:A", 28)
        writer.sheets["Summary"].set_column("B:D", 18, money_fmt)

        dr = writer.sheets["Daily_Results"]
        dr.set_column("A:A", 14, date_fmt)
        dr.set_column("B:D", 14, pct_fmt)
        dr.set_column("E:G", 18, money_fmt)
        dr.set_column("H:I", 14, pct_fmt)
        dr.set_column("J:J", 14, pct_fmt)
        dr.set_column("K:N", 12)

        mr = writer.sheets["Monthly_Results"]
        mr.set_column("A:A", 14, date_fmt)
        mr.set_column("B:D", 18, money_fmt)
        mr.set_column("E:G", 14, pct_fmt)

        mp = writer.sheets["Monthly_Picks"]
        mp.set_column("A:A", 14, date_fmt)
        mp.set_column("B:B", 18)
        mp.set_column("C:C", 12, pct_fmt)
        mp.set_column("D:G", 14)

        tr = writer.sheets["Trades"]
        tr.set_column("A:A", 14, date_fmt)
        tr.set_column("B:B", 18)
        tr.set_column("C:C", 12)
        tr.set_column("D:D", 14, money_fmt)
        tr.set_column("E:E", 12, pct_fmt)
        tr.set_column("F:F", 24)

        if "Roundtrip_Trades" in writer.sheets:
            rt = writer.sheets["Roundtrip_Trades"]
            rt.set_column("A:A", 18)
            rt.set_column("B:C", 14, date_fmt)
            rt.set_column("D:E", 14, money_fmt)
            rt.set_column("F:F", 12, pct_fmt)
            rt.set_column("G:G", 10)
            rt.set_column("H:H", 20)

        rg = writer.sheets["Regime"]
        rg.set_column("A:A", 14, date_fmt)
        rg.set_column("B:B", 10)
        rg.set_column("C:C", 12, pct_fmt)
        rg.set_column("D:D", 12, pct_fmt)
        rg.set_column("E:H", 14)

        chart = workbook.add_chart({"type": "line"})
        rows = len(bt["monthly_results"])
        chart.add_series({
            "name": "Strategy",
            "categories": ["Monthly_Results", 1, 0, rows, 0],
            "values": ["Monthly_Results", 1, 1, rows, 1],
        })
        chart.add_series({
            "name": "Nifty 200",
            "categories": ["Monthly_Results", 1, 0, rows, 0],
            "values": ["Monthly_Results", 1, 2, rows, 2],
        })
        if has_mom:
            chart.add_series({
                "name": "Nifty 200 Mom 30",
                "categories": ["Monthly_Results", 1, 0, rows, 0],
                "values": ["Monthly_Results", 1, 3, rows, 3],
            })
        chart.set_title({"name": "Strategy vs Benchmarks"})
        chart.set_x_axis({"name": "Month"})
        chart.set_y_axis({"name": "Portfolio Value"})
        chart.set_size({"width": 900, "height": 420})
        mr.insert_chart("I2", chart)

    logger.info("Excel written: %s", output_file)


# ---------------------------------------------------------------------------
# CLI compatibility wrapper
# ---------------------------------------------------------------------------

def run_backtest_from_config(yaml_path: str | Path | None = None) -> dict:
    """Run the full momentum backtest using config.yaml parameters.

    Returns the result dict so callers (CLI, API) can inspect or export.
    """
    config = load_config_from_yaml(yaml_path)
    bt = run_backtest(config)

    summary = bt["summary"]
    strat_cagr = summary.loc[summary["Metric"] == "CAGR", "Strategy"].iloc[0]
    strat_mdd = summary.loc[summary["Metric"] == "Max Drawdown", "Strategy"].iloc[0]
    strat_vol = summary.loc[summary["Metric"] == "Annualized Volatility", "Strategy"].iloc[0]
    strat_sharpe = summary.loc[summary["Metric"] == "Sharpe (rf=4%)", "Strategy"].iloc[0]

    logger.info("Backtest complete:")
    logger.info("  CAGR:       %.2f%%", strat_cagr * 100)
    logger.info("  Max DD:     %.2f%%", strat_mdd * 100)
    logger.info("  Volatility: %.2f%%", strat_vol * 100)
    logger.info("  Sharpe:     %.2f", strat_sharpe)

    print(f"\nBacktest Results (config.yaml)")
    print(f"  CAGR:       {strat_cagr:.2%}")
    print(f"  Max DD:     {strat_mdd:.2%}")
    print(f"  Volatility: {strat_vol:.2%}")
    print(f"  Sharpe:     {strat_sharpe:.2f}")

    export_to_excel(bt, config.output_file)
    print(f"\n  Full results exported to {config.output_file}")

    return bt


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_backtest_from_config()
