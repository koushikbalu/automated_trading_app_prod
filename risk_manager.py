"""Risk management: circuit breaker, sector caps, minimum exposure floor,
and pre-trade validation.

Ported from the risk logic in momentum_trading_backtest.py (lines 486-510,
710-742).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from constants import SECTOR_MAP
from models import OrderAction, Position, RegimeLevel, RegimeState, Signal
from utils import capped_inverse_vol_weights

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Drawdown circuit breaker with market-based reset.

    Triggers at configurable portfolio drawdown from peak. Liquidates all
    positions.  Resets only after N consecutive full-risk-on days, at which
    point the peak is also reset.
    """

    def __init__(self, threshold: float = -0.20, reset_days: int = 5):
        self.threshold = threshold
        self.reset_days = reset_days
        self.active = False
        self._risk_on_streak = 0

    def check(
        self,
        portfolio_value: float,
        portfolio_peak: float,
        regime: RegimeState,
    ) -> bool:
        """Return True if circuit breaker should trigger or remain active."""
        if self.active:
            if regime.level == RegimeLevel.FULL_RISK_ON:
                self._risk_on_streak += 1
            else:
                self._risk_on_streak = 0
            if self._risk_on_streak >= self.reset_days:
                self.active = False
                self._risk_on_streak = 0
                logger.info(
                    "Circuit breaker RESET (%d consecutive risk-on days, peak reset to %.0f)",
                    self.reset_days,
                    portfolio_value,
                )
                return False
            return True

        dd = (portfolio_value / portfolio_peak) - 1 if portfolio_peak > 0 else 0.0
        if dd < self.threshold:
            self.active = True
            self._risk_on_streak = 0
            logger.warning("Circuit breaker TRIGGERED (DD: %.1f%%)", dd * 100)
            return True
        return False


# ---------------------------------------------------------------------------
# Sector cap validation
# ---------------------------------------------------------------------------

def enforce_sector_caps(
    target_weights: dict[str, float],
    max_sector_weight: float = 0.30,
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Scale down weights if any sector exceeds its cap, redistributing
    the excess proportionally among under-cap sectors.
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    sector_totals: dict[str, float] = {}
    for ticker, w in target_weights.items():
        sec = sector_map.get(ticker, "Other")
        sector_totals[sec] = sector_totals.get(sec, 0) + w

    over_sectors = {s: t for s, t in sector_totals.items() if t > max_sector_weight}
    if not over_sectors:
        return target_weights

    adjusted = dict(target_weights)
    for sector, total in over_sectors.items():
        scale = max_sector_weight / total
        for ticker in list(adjusted.keys()):
            if sector_map.get(ticker, "Other") == sector:
                adjusted[ticker] *= scale

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}

    return adjusted


# ---------------------------------------------------------------------------
# Minimum exposure floor
# ---------------------------------------------------------------------------

def apply_exposure_floor(
    current_positions: dict[str, Position],
    ranked_candidates: pd.DataFrame,
    stocks_close: pd.DataFrame,
    atr_df: pd.DataFrame,
    cfg: dict | None = None,
) -> list[Signal]:
    """When regime allows allocation but exposure < min_exposure, fill up to
    min_exposure_slots from top-ranked candidates not already held.
    """
    if cfg is None:
        cfg = _load_config()
    risk = cfg.get("risk", {})
    sizing = cfg.get("sizing", {})
    exits_cfg = cfg.get("exits", {})
    min_exp = risk.get("min_exposure", 0.30)
    slots = risk.get("min_exposure_slots", 3)
    min_price = cfg.get("strategy", {}).get("min_price", 100.0)
    max_weight = sizing.get("max_weight_per_stock", 0.18)

    pos_exposure = sum(p.weight for p in current_positions.values())
    if pos_exposure >= min_exp or ranked_candidates.empty:
        return []

    needed = min_exp - pos_exposure
    already_held = set(current_positions.keys())

    floor_candidates = [
        t for t in ranked_candidates.index
        if t not in already_held
        and t in stocks_close.columns
        and pd.notna(stocks_close[t].iloc[-1])
        and stocks_close[t].iloc[-1] >= min_price
    ][:slots]

    if not floor_candidates:
        return []

    floor_vols = ranked_candidates.loc[floor_candidates, "vol"]
    floor_weights = capped_inverse_vol_weights(floor_vols, max_weight)
    total_fw = floor_weights.sum()
    if total_fw > 0:
        floor_weights = floor_weights / total_fw * needed

    signals: list[Signal] = []
    for ticker in floor_candidates:
        fw = float(floor_weights.loc[ticker])
        price = float(stocks_close[ticker].iloc[-1])
        signals.append(Signal(
            ticker=ticker,
            action=OrderAction.BUY,
            target_weight=fw,
            current_weight=0.0,
            price=price,
            reason="Min exposure floor",
        ))

    return signals


# ---------------------------------------------------------------------------
# Pre-trade validation
# ---------------------------------------------------------------------------

def validate_order(
    ticker: str,
    weight: float,
    portfolio_value: float,
    available_capital: float,
    current_position_count: int,
    max_positions: int = 10,
) -> tuple[bool, str]:
    """Pre-trade checks: capital sufficiency, position limits."""
    order_value = weight * portfolio_value

    if order_value > available_capital:
        return False, f"Insufficient capital: need {order_value:.0f}, have {available_capital:.0f}"

    if current_position_count >= max_positions:
        return False, f"Max positions ({max_positions}) reached"

    if order_value < 1000:
        return False, f"Order value too small: {order_value:.0f}"

    return True, ""
