"""Rebalance signal generation: graduated regime filter, blended momentum
scoring, inverse-vol target portfolio, and diff vs current holdings.

Ported from the rebalance block (lines 549-670) of
momentum_trading_backtest.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from constants import SECTOR_MAP
from models import (
    OrderAction,
    Position,
    RegimeLevel,
    RegimeState,
    RebalanceResult,
    Signal,
)
from utils import (
    adv_126,
    blended_weights,
    capped_inverse_vol_weights,
    compute_atr_df,
    rolling_volatility,
    sma,
)

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Regime assessment
# ---------------------------------------------------------------------------

def assess_regime(
    benchmark_close: pd.Series,
    stocks_close: pd.DataFrame,
    cfg: dict | None = None,
) -> RegimeState:
    """Graduated regime filter: full risk-on / neutral / risk-off.

    Full risk-on (100%): bench > 200DMA AND golden cross AND
        3-month return > 0 AND breadth > 40%.
    Neutral (50%): bench > 200DMA AND breadth > 30%.
    Risk-off (0%): neither condition met.
    """
    if cfg is None:
        cfg = _load_config()
    regime_cfg = cfg.get("regime", {})
    breadth_thresh = regime_cfg.get("breadth_threshold", 0.40)
    neutral_thresh = regime_cfg.get("neutral_breadth_threshold", 0.30)
    neutral_alloc = regime_cfg.get("neutral_allocation_pct", 0.50)
    require_3m = regime_cfg.get("require_positive_3m_return", True)

    bench_200 = sma(benchmark_close, 200)
    bench_50 = sma(benchmark_close, 50)
    bench_63ret = benchmark_close / benchmark_close.shift(63) - 1

    dma_200 = sma(stocks_close, 200)
    current = stocks_close.iloc[-1]
    above_200 = current > dma_200.iloc[-1]
    notna_count = above_200.notna().sum()
    breadth = float(above_200.sum() / notna_count) if notna_count > 0 else 0.0

    b_close = float(benchmark_close.iloc[-1])
    b_200 = float(bench_200.iloc[-1]) if pd.notna(bench_200.iloc[-1]) else 0.0
    b_50 = float(bench_50.iloc[-1]) if pd.notna(bench_50.iloc[-1]) else 0.0
    b_3m = float(bench_63ret.iloc[-1]) if pd.notna(bench_63ret.iloc[-1]) else 0.0

    bench_above_200 = b_close > b_200 and b_200 > 0

    full_risk_on = (
        bench_above_200
        and (b_50 > b_200)
        and (not require_3m or b_3m > 0)
        and (breadth > breadth_thresh)
    )
    neutral = (
        not full_risk_on
        and bench_above_200
        and (breadth > neutral_thresh)
    )

    if full_risk_on:
        level = RegimeLevel.FULL_RISK_ON
        alloc = 1.0
    elif neutral:
        level = RegimeLevel.NEUTRAL
        alloc = neutral_alloc
    else:
        level = RegimeLevel.RISK_OFF
        alloc = 0.0

    return RegimeState(
        level=level,
        allocation_pct=alloc,
        breadth=breadth,
        bench_close=b_close,
        bench_50dma=b_50,
        bench_200dma=b_200,
        bench_3m_return=b_3m,
    )


# ---------------------------------------------------------------------------
# Momentum scoring & candidate selection
# ---------------------------------------------------------------------------

def score_and_rank(
    stocks_close: pd.DataFrame,
    stocks_volume: pd.DataFrame,
    vol_60: pd.DataFrame,
    adv: pd.DataFrame,
    dma_100: pd.DataFrame,
    dma_200: pd.DataFrame,
    sector_map: dict[str, str],
    cfg: dict,
) -> pd.DataFrame:
    """Score universe, apply filters, return ranked candidate DataFrame."""
    strat = cfg.get("strategy", {})
    top_liquid_n = strat.get("top_liquid_n", 120)
    min_price = strat.get("min_price", 100.0)
    weights = strat.get("momentum_weights", [0.4, 0.3, 0.2, 0.1])

    current_prices = stocks_close.iloc[-1]

    liquid = adv.iloc[-1].dropna().sort_values(ascending=False)
    liquid = liquid[current_prices.reindex(liquid.index) >= min_price]
    liquid_universe = liquid.head(top_liquid_n).index.tolist()

    if not liquid_universe:
        return pd.DataFrame()

    lu = liquid_universe
    ret_1m = current_prices[lu] / stocks_close.shift(21).iloc[-1][lu] - 1
    ret_3m = current_prices[lu] / stocks_close.shift(63).iloc[-1][lu] - 1
    ret_6m = current_prices[lu] / stocks_close.shift(126).iloc[-1][lu] - 1
    ret_12m = stocks_close.shift(21).iloc[-1][lu] / stocks_close.shift(252).iloc[-1][lu] - 1

    raw_mom = weights[0] * ret_12m + weights[1] * ret_6m + weights[2] * ret_3m + weights[3] * ret_1m

    elig = (
        (current_prices[lu] > dma_100.iloc[-1][lu])
        & (current_prices[lu] > dma_200.iloc[-1][lu])
    )

    vol_20 = stocks_volume.iloc[-20:].mean()
    vol_60_avg = stocks_volume.iloc[-60:].mean()
    rel_volume = (vol_20 / vol_60_avg.replace(0, np.nan)).reindex(lu)
    volume_ok = rel_volume >= rel_volume.median()

    cand = pd.DataFrame({
        "score": raw_mom,
        "vol": vol_60.iloc[-1].reindex(lu),
        "eligible": elig,
        "volume_ok": volume_ok,
    }).dropna()

    cand = cand[cand["eligible"] & cand["volume_ok"]]
    max_vol_pct = strat.get("max_vol_percentile", 1.0)
    if max_vol_pct < 1.0 and not cand.empty:
        vol_cap = cand["vol"].quantile(max_vol_pct)
        cand = cand[cand["vol"] <= vol_cap]
    cand["risk_adj_score"] = cand["score"] / cand["vol"].replace(0, np.nan)
    cand = cand.dropna()

    penalty = strat.get("sector_downtrend_penalty", 1.0)
    if penalty < 1.0:
        cand["_sector"] = cand.index.map(lambda t: sector_map.get(t, "Other"))
        sector_median_mom = cand.groupby("_sector")["score"].median()
        bad_sectors = set(sector_median_mom[sector_median_mom < 0].index)
        if bad_sectors:
            mask = cand["_sector"].isin(bad_sectors)
            cand.loc[mask, "risk_adj_score"] *= penalty
        cand = cand.drop(columns=["_sector"])

    cand = cand.sort_values("risk_adj_score", ascending=False)

    return cand


def select_with_sector_caps(
    cand: pd.DataFrame,
    num_slots: int,
    sector_map: dict[str, str],
    max_sector_weight: float,
) -> list[str]:
    """Pick top candidates while enforcing sector diversification."""
    max_per_sector = max(1, int(num_slots * max_sector_weight) + 1)
    sector_counts: dict[str, int] = {}
    selected: list[str] = []
    for t in cand.index:
        sec = sector_map.get(t, "Other")
        if sector_counts.get(sec, 0) < max_per_sector:
            selected.append(t)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) >= num_slots:
            break
    return selected


# ---------------------------------------------------------------------------
# Full rebalance signal generation
# ---------------------------------------------------------------------------

def generate_rebalance_signals(
    stocks_close: pd.DataFrame,
    stocks_high: pd.DataFrame,
    stocks_low: pd.DataFrame,
    stocks_volume: pd.DataFrame,
    benchmark_close: pd.Series,
    current_positions: dict[str, Position],
    cfg: dict | None = None,
) -> RebalanceResult:
    """Produce a full rebalance signal: regime -> score -> weight -> diff."""
    if cfg is None:
        cfg = _load_config()

    strat = cfg.get("strategy", {})
    sizing = cfg.get("sizing", {})
    top_n = strat.get("top_momentum_n", 10)
    max_weight = sizing.get("max_weight_per_stock", 0.18)
    min_weight = sizing.get("min_weight_per_stock", 0.04)
    score_blend = sizing.get("score_blend", 0.4)
    max_sector_weight = sizing.get("max_sector_weight", 0.30)
    vol_lookback = sizing.get("volatility_lookback_days", 60)

    regime = assess_regime(benchmark_close, stocks_close, cfg)
    num_slots = int(round(top_n * regime.allocation_pct))

    daily_returns = stocks_close.pct_change().fillna(0)
    vol_60 = rolling_volatility(daily_returns, vol_lookback)
    adv = adv_126(stocks_close, stocks_volume)
    dma_100 = sma(stocks_close, 100)
    dma_200 = sma(stocks_close, 200)

    buys: list[Signal] = []
    sells: list[Signal] = []
    target_weights: dict[str, float] = {}

    if num_slots > 0:
        cand = score_and_rank(
            stocks_close, stocks_volume, vol_60, adv,
            dma_100, dma_200, SECTOR_MAP, cfg,
        )
        if not cand.empty:
            bonus = strat.get("continuity_bonus", 0.5)
            if bonus > 0:
                for t in current_positions:
                    if t in cand.index:
                        cand.loc[t, "risk_adj_score"] *= (1 + bonus)
                cand = cand.sort_values("risk_adj_score", ascending=False)

            selected = select_with_sector_caps(
                cand, num_slots, SECTOR_MAP, max_sector_weight,
            )
            if selected:
                new_weights = blended_weights(
                    cand.loc[selected, "vol"],
                    cand.loc[selected, "risk_adj_score"],
                    max_weight,
                    score_blend,
                )
                keep = new_weights[new_weights >= min_weight]
                if not keep.empty:
                    new_weights = keep / keep.sum()
                    selected = list(keep.index)
                    target_weights = {t: float(new_weights.loc[t]) for t in selected}
                else:
                    target_weights = {}
            else:
                target_weights = {}
    else:
        selected = []

    existing = set(current_positions.keys())
    target_set = set(target_weights.keys())

    to_exit = existing - target_set
    for ticker in to_exit:
        pos = current_positions[ticker]
        price = float(stocks_close[ticker].iloc[-1]) if ticker in stocks_close.columns else pos.entry_price
        sells.append(Signal(
            ticker=ticker,
            action=OrderAction.REBAL_EXIT,
            target_weight=0.0,
            current_weight=pos.weight,
            price=price,
            reason="Dropped from target universe",
        ))

    for ticker, tw in target_weights.items():
        price = float(stocks_close[ticker].iloc[-1]) if ticker in stocks_close.columns else 0.0
        cw = current_positions[ticker].weight if ticker in current_positions else 0.0
        if ticker not in existing:
            buys.append(Signal(
                ticker=ticker,
                action=OrderAction.BUY,
                target_weight=tw,
                current_weight=0.0,
                price=price,
                reason="Monthly rebalance entry",
            ))
        elif abs(tw - cw) > 0.005:
            if tw > cw:
                buys.append(Signal(
                    ticker=ticker,
                    action=OrderAction.BUY,
                    target_weight=tw,
                    current_weight=cw,
                    price=price,
                    reason="Weight increase",
                ))
            else:
                sells.append(Signal(
                    ticker=ticker,
                    action=OrderAction.SELL,
                    target_weight=tw,
                    current_weight=cw,
                    price=price,
                    reason="Weight decrease",
                ))

    return RebalanceResult(
        date=datetime.now(),
        regime=regime,
        num_selected=len(target_weights),
        buys=buys,
        sells=sells,
        target_weights=target_weights,
    )
