"""Shared fixtures for the automated trading test suite."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Position, PortfolioSnapshot, Signal, OrderAction, TradeRecord


# ---------------------------------------------------------------------------
# Sample config (mirrors config.yaml structure)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config() -> dict:
    return {
        "strategy": {
            "universe_file": "universe.csv",
            "benchmark": "NSE:NIFTY 200",
            "top_liquid_n": 120,
            "top_momentum_n": 10,
            "min_price": 50.0,
            "momentum_weights": [0.25, 0.40, 0.25, 0.10],
            "continuity_bonus": 0.3,
            "sector_downtrend_penalty": 1.0,
            "max_vol_percentile": 1.0,
        },
        "regime": {
            "breadth_threshold": 0.30,
            "neutral_breadth_threshold": 0.25,
            "neutral_allocation_pct": 0.70,
            "require_bench_above_200dma": True,
            "require_golden_cross": True,
            "require_positive_3m_return": False,
        },
        "sizing": {
            "method": "inverse_volatility",
            "max_weight_per_stock": 0.18,
            "min_weight_per_stock": 0.04,
            "score_blend": 0.4,
            "max_sector_weight": 0.25,
            "volatility_lookback_days": 60,
        },
        "exits": {
            "use_atr_trailing_stop": True,
            "atr_window": 14,
            "atr_multiple": 2.0,
            "max_loss_pct": -0.08,
            "max_hold_losing_days": 0,
            "stop_exit_slippage": 0.005,
            "use_50dma_exit": False,
        },
        "pyramid": {
            "enabled": True,
            "threshold_pct": 0.03,
            "add_pct": 0.03,
            "max_pyramids": 2,
            "ratchet_stop_to_breakeven": True,
        },
        "risk": {
            "drawdown_circuit_breaker": -0.20,
            "cb_reset_days": 1,
            "min_exposure": 0.30,
            "min_exposure_slots": 3,
            "re_entry_enabled": True,
        },
        "costs": {
            "one_way_brokerage": 0.0003,
            "stt_sell_side": 0.001,
            "gst_rate": 0.18,
            "stamp_duty_buy": 0.00015,
            "slippage_estimate": 0.001,
        },
        "capital": {"initial": 20_000_000},
        "broker": {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "access_token_file": "kite_token.json",
            "exchange": "NSE",
            "product_type": "CNC",
            "order_type": "MARKET",
            "limit_buffer_pct": 0.002,
            "dry_run": True,
        },
        "execution": {
            "max_participation_rate": 0.05,
            "max_slice_value": 2_500_000,
            "slice_delay_seconds": 0,
            "freeze_qty_buffer": 0.90,
        },
        "notifications": {
            "telegram": {
                "bot_token": "",
                "chat_id": "",
                "send_on": [],
            }
        },
        "persistence": {"database_url": "sqlite:///:memory:"},
        "schedule": {},
    }


# ---------------------------------------------------------------------------
# Price data helpers
# ---------------------------------------------------------------------------

def _random_walk(start: float, n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = start * np.cumprod(1 + returns)
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    return pd.Series(prices, index=dates, name="price")


@pytest.fixture()
def price_data() -> dict[str, pd.DataFrame]:
    """Generate 300-day synthetic OHLCV for 15 stocks + benchmark."""
    n = 300
    tickers = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE",
    ]
    close_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}
    for i, t in enumerate(tickers):
        c = _random_walk(1000 + i * 200, n, seed=42 + i)
        close_dict[t] = c
        high_dict[t] = c * 1.01
        low_dict[t] = c * 0.99
        vol_dict[t] = pd.Series(
            np.random.default_rng(i).integers(100_000, 1_000_000, n),
            index=c.index, dtype=float,
        )

    bench = _random_walk(15000, n, seed=99)
    close_dict["NIFTY 200"] = bench
    high_dict["NIFTY 200"] = bench * 1.005
    low_dict["NIFTY 200"] = bench * 0.995
    vol_dict["NIFTY 200"] = pd.Series(0.0, index=bench.index)

    return {
        "close": pd.DataFrame(close_dict),
        "high": pd.DataFrame(high_dict),
        "low": pd.DataFrame(low_dict),
        "volume": pd.DataFrame(vol_dict),
    }


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_positions() -> dict[str, Position]:
    base = datetime.now() - timedelta(days=30)
    return {
        "RELIANCE": Position(
            ticker="RELIANCE", weight=0.12, entry_price=2500,
            entry_date=base, high_watermark=2700, stop_price=2400,
            sector="Energy",
        ),
        "TCS": Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=base, high_watermark=3600, stop_price=3200,
            sector="IT",
        ),
        "HDFCBANK": Position(
            ticker="HDFCBANK", weight=0.08, entry_price=1600,
            entry_date=base, high_watermark=1700, stop_price=1500,
            sector="Banking",
        ),
    }


@pytest.fixture()
def sample_snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        date=datetime.now(),
        portfolio_value=20_000_000,
        portfolio_peak=20_500_000,
        circuit_breaker_active=False,
        cash_weight=0.70,
        exposure=0.30,
        positions_count=3,
    )
