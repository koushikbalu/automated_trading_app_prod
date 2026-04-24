"""ATR trailing stop management, high-watermark tracking, and intra-month
re-entry logic.

Ported from momentum_trading_backtest.py lines 760-785 (stops) and
680-707 (re-entry).
"""

from __future__ import annotations

import logging
from datetime import date as _date_type
from typing import Optional

import pandas as pd
import yaml
from pathlib import Path

from models import OrderAction, Position, Signal
from utils import compute_atr_series, sma

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ATR trailing stop
# ---------------------------------------------------------------------------

def compute_initial_stop(
    entry_price: float,
    atr_value: float,
    atr_multiple: float = 2.5,
) -> Optional[float]:
    """Initial stop = entry_price - atr_multiple * ATR."""
    if pd.isna(atr_value) or atr_value <= 0:
        return None
    return entry_price - atr_multiple * atr_value


def update_trailing_stop(
    position: Position,
    current_price: float,
    atr_value: float,
    atr_multiple: float = 2.5,
) -> Position:
    """Update high watermark and ratchet stop upward. Never lowers the stop."""
    position.high_watermark = max(position.high_watermark, current_price)

    if pd.notna(atr_value) and atr_value > 0:
        candidate_stop = position.high_watermark - atr_multiple * atr_value
        if position.stop_price is None:
            position.stop_price = candidate_stop
        else:
            position.stop_price = max(position.stop_price, candidate_stop)

    return position


def check_stop_triggered(
    position: Position,
    current_price: float,
) -> tuple[bool, str]:
    """Return (triggered, reason) if current price breaches the stop."""
    if position.stop_price is not None and current_price < position.stop_price:
        return True, "ATR Trailing Stop"
    return False, ""


# ---------------------------------------------------------------------------
# Batch stop check for all positions
# ---------------------------------------------------------------------------

def check_all_stops(
    positions: dict[str, Position],
    live_prices: dict[str, float],
    atr_values: dict[str, float],
    cfg: dict | None = None,
) -> tuple[list[Signal], dict[str, Position]]:
    """Check ATR stops for all positions against live prices.

    Returns:
        exits: list of sell Signals for triggered stops
        updated_positions: positions with updated watermarks/stops
    """
    if cfg is None:
        cfg = _load_config()
    exits_cfg = cfg.get("exits", {})
    atr_mult = exits_cfg.get("atr_multiple", 2.0)
    max_loss_pct = exits_cfg.get("max_loss_pct", -0.08)
    max_losing_days = exits_cfg.get("max_hold_losing_days", 10)

    exits: list[Signal] = []

    for ticker, pos in positions.items():
        price = live_prices.get(ticker)
        if price is None:
            continue

        atr_val = atr_values.get(ticker, 0.0)
        update_trailing_stop(pos, price, atr_val, atr_mult)

        triggered, reason = check_stop_triggered(pos, price)

        if not triggered and max_loss_pct is not None:
            pnl_pct = (price / pos.entry_price) - 1 if pos.entry_price > 0 else 0.0
            if pnl_pct < max_loss_pct:
                triggered = True
                reason = f"Hard stop ({max_loss_pct:.0%})"

        today = _date_type.today()
        last_check = getattr(pos, "_last_losing_check", None)
        if last_check != today:
            if price < pos.entry_price:
                pos.losing_days += 1
            else:
                pos.losing_days = 0
            pos._last_losing_check = today

        if max_losing_days > 0 and not triggered and pos.losing_days >= max_losing_days:
            triggered = True
            reason = f"Time stop ({max_losing_days}d underwater)"

        if triggered:
            exits.append(Signal(
                ticker=ticker,
                action=OrderAction.SELL,
                target_weight=0.0,
                current_weight=pos.weight,
                price=price,
                reason=reason,
            ))
            logger.info(
                "STOP triggered: %s at %.2f (reason: %s)",
                ticker, price, reason,
            )

    return exits, positions


# ---------------------------------------------------------------------------
# Intra-month re-entry
# ---------------------------------------------------------------------------

def check_re_entry(
    stopped_out: dict[str, float],
    ranked_candidates: pd.DataFrame,
    live_prices: dict[str, float],
    dma_20: dict[str, float],
    atr_values: dict[str, float],
    current_positions: dict[str, Position],
    top_n: int = 10,
    cfg: dict | None = None,
) -> list[Signal]:
    """Check if any stopped-out stock qualifies for re-entry.

    Conditions:
    - Price recovered above 20DMA
    - Stock remains in top momentum ranking
    """
    if cfg is None:
        cfg = _load_config()
    re_entry_enabled = cfg.get("risk", {}).get("re_entry_enabled", True)
    if not re_entry_enabled or not stopped_out:
        return []

    top_ranked = set(ranked_candidates.head(top_n).index) if not ranked_candidates.empty else set()
    signals: list[Signal] = []

    for ticker, orig_weight in list(stopped_out.items()):
        if ticker in current_positions:
            continue

        price = live_prices.get(ticker)
        ma20 = dma_20.get(ticker)
        if price is None or ma20 is None:
            continue

        if price > ma20 and ticker in top_ranked:
            signals.append(Signal(
                ticker=ticker,
                action=OrderAction.BUY,
                target_weight=orig_weight,
                current_weight=0.0,
                price=price,
                reason="Re-entry after ATR stop",
            ))
            logger.info(
                "RE-ENTRY qualified: %s at %.2f (above 20DMA %.2f)",
                ticker, price, ma20,
            )

    return signals
