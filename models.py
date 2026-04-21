"""Dataclasses used across the trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RegimeLevel(Enum):
    FULL_RISK_ON = "full_risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    REBAL_EXIT = "REBAL_EXIT"


@dataclass
class RegimeState:
    level: RegimeLevel
    allocation_pct: float
    breadth: float
    bench_close: float
    bench_50dma: float
    bench_200dma: float
    bench_3m_return: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    ticker: str
    weight: float
    entry_price: float
    entry_date: datetime
    high_watermark: float
    stop_price: Optional[float] = None
    current_value: Optional[float] = None
    sector: str = "Other"
    losing_days: int = 0
    pyramid_count: int = 0


@dataclass
class Signal:
    ticker: str
    action: OrderAction
    target_weight: float
    current_weight: float
    price: float
    reason: str


@dataclass
class TradeRecord:
    date: datetime
    ticker: str
    action: str
    price: float
    weight_traded: float
    reason: str
    order_id: Optional[str] = None
    costs: float = 0.0
    holding_days: Optional[int] = None
    pnl_pct: Optional[float] = None


@dataclass
class RebalanceResult:
    date: datetime
    regime: RegimeState
    num_selected: int
    buys: list[Signal] = field(default_factory=list)
    sells: list[Signal] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    date: datetime
    portfolio_value: float
    portfolio_peak: float
    circuit_breaker_active: bool
    cash_weight: float
    exposure: float
    positions_count: int
