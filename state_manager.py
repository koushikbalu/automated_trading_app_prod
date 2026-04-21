"""PostgreSQL persistence via SQLAlchemy 2.0.

Tables: positions, trades, portfolio_state, regime_history,
rebalance_history, stopped_out_tracker.

On restart the system loads state from DB and resumes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from typing import Optional

import yaml
from pathlib import Path
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from models import Position, PortfolioSnapshot, RegimeState, TradeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM base and table definitions
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class PositionRow(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, unique=True, index=True)
    weight = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_date = Column(DateTime, nullable=False)
    high_watermark = Column(Float, nullable=False)
    stop_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    sector = Column(String(32), nullable=False, default="Other")
    status = Column(String(16), nullable=False, default="open")
    losing_days = Column(Integer, nullable=False, default=0)
    pyramid_count = Column(Integer, nullable=False, default=0)


class TradeRow(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    action = Column(String(16), nullable=False)
    price = Column(Float, nullable=False)
    weight_traded = Column(Float, nullable=False)
    reason = Column(String(128), nullable=True)
    order_id = Column(String(64), nullable=True)
    costs = Column(Float, nullable=False, default=0.0)
    holding_days = Column(Integer, nullable=True)
    pnl_pct = Column(Float, nullable=True)


class PortfolioStateRow(Base):
    __tablename__ = "portfolio_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    portfolio_value = Column(Float, nullable=False)
    portfolio_peak = Column(Float, nullable=False)
    circuit_breaker_active = Column(Boolean, nullable=False, default=False)
    cash_weight = Column(Float, nullable=False, default=1.0)
    exposure = Column(Float, nullable=False, default=0.0)
    positions_count = Column(Integer, nullable=False, default=0)


class RegimeHistoryRow(Base):
    __tablename__ = "regime_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    level = Column(String(20), nullable=False)
    allocation_pct = Column(Float, nullable=False)
    breadth = Column(Float, nullable=False)
    bench_close = Column(Float, nullable=False)
    bench_50dma = Column(Float, nullable=False)
    bench_200dma = Column(Float, nullable=False)
    bench_3m_return = Column(Float, nullable=False)


class RebalanceHistoryRow(Base):
    __tablename__ = "rebalance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rebalance_date = Column(DateTime, nullable=False, index=True)
    num_selected = Column(Integer, nullable=False)
    allocation_pct = Column(Float, nullable=False)
    picks_json = Column(Text, nullable=True)


class StoppedOutRow(Base):
    __tablename__ = "stopped_out_tracker"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    weight = Column(Float, nullable=False)
    stop_date = Column(DateTime, nullable=False)
    month_year = Column(String(7), nullable=False)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

def _resolve_db_url() -> str:
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get("persistence", {}).get("database_url", "")
        if raw.startswith("${") and raw.endswith("}"):
            return os.environ.get(raw[2:-1], "sqlite:///momentum_state.db")
        if raw:
            return raw
    return os.environ.get("DATABASE_URL", "sqlite:///momentum_state.db")


class StateManager:
    """CRUD operations for all persistent state."""

    def __init__(self, db_url: str | None = None):
        url = db_url or _resolve_db_url()
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def _session(self) -> Session:
        return self._Session()

    # -- Positions ---------------------------------------------------------

    def load_positions(self) -> dict[str, Position]:
        with self._session() as s:
            rows = s.query(PositionRow).filter(PositionRow.status == "open").all()
            return {
                r.ticker: Position(
                    ticker=r.ticker,
                    weight=r.weight,
                    entry_price=r.entry_price,
                    entry_date=r.entry_date,
                    high_watermark=r.high_watermark,
                    stop_price=r.stop_price,
                    sector=r.sector,
                    losing_days=r.losing_days or 0,
                    pyramid_count=r.pyramid_count or 0,
                )
                for r in rows
            }

    def save_position(self, pos: Position) -> None:
        with self._session() as s:
            existing = s.query(PositionRow).filter(PositionRow.ticker == pos.ticker).first()
            if existing:
                existing.weight = pos.weight
                existing.entry_price = pos.entry_price
                existing.entry_date = pos.entry_date
                existing.high_watermark = pos.high_watermark
                existing.stop_price = pos.stop_price
                existing.sector = pos.sector
                existing.status = "open"
                existing.losing_days = pos.losing_days
                existing.pyramid_count = pos.pyramid_count
            else:
                s.add(PositionRow(
                    ticker=pos.ticker,
                    weight=pos.weight,
                    entry_price=pos.entry_price,
                    entry_date=pos.entry_date,
                    high_watermark=pos.high_watermark,
                    stop_price=pos.stop_price,
                    sector=pos.sector,
                    status="open",
                    losing_days=pos.losing_days,
                    pyramid_count=pos.pyramid_count,
                ))
            s.commit()

    def close_position(self, ticker: str) -> None:
        with self._session() as s:
            row = s.query(PositionRow).filter(PositionRow.ticker == ticker).first()
            if row:
                row.status = "closed"
                s.commit()

    def update_position_prices(self, prices: dict[str, float]) -> None:
        with self._session() as s:
            for ticker, price in prices.items():
                row = s.query(PositionRow).filter(
                    PositionRow.ticker == ticker, PositionRow.status == "open"
                ).first()
                if row:
                    row.current_price = price
            s.commit()

    # -- Trades ------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        with self._session() as s:
            s.add(TradeRow(
                date=trade.date,
                ticker=trade.ticker,
                action=trade.action,
                price=trade.price,
                weight_traded=trade.weight_traded,
                reason=trade.reason,
                order_id=trade.order_id,
                costs=trade.costs,
                holding_days=trade.holding_days,
                pnl_pct=trade.pnl_pct,
            ))
            s.commit()

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        with self._session() as s:
            rows = (
                s.query(TradeRow)
                .order_by(TradeRow.date.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "date": r.date.isoformat() if r.date else None,
                    "ticker": r.ticker,
                    "action": r.action,
                    "price": r.price,
                    "weight_traded": r.weight_traded,
                    "reason": r.reason,
                    "order_id": r.order_id,
                    "costs": r.costs,
                }
                for r in rows
            ]

    # -- Portfolio state ---------------------------------------------------

    def save_portfolio_state(self, snap: PortfolioSnapshot) -> None:
        with self._session() as s:
            s.add(PortfolioStateRow(
                date=snap.date,
                portfolio_value=snap.portfolio_value,
                portfolio_peak=snap.portfolio_peak,
                circuit_breaker_active=snap.circuit_breaker_active,
                cash_weight=snap.cash_weight,
                exposure=snap.exposure,
                positions_count=snap.positions_count,
            ))
            s.commit()

    def load_latest_portfolio_state(self) -> Optional[PortfolioSnapshot]:
        with self._session() as s:
            row = (
                s.query(PortfolioStateRow)
                .order_by(PortfolioStateRow.date.desc())
                .first()
            )
            if not row:
                return None
            return PortfolioSnapshot(
                date=row.date,
                portfolio_value=row.portfolio_value,
                portfolio_peak=row.portfolio_peak,
                circuit_breaker_active=row.circuit_breaker_active,
                cash_weight=row.cash_weight,
                exposure=row.exposure,
                positions_count=row.positions_count,
            )

    # -- Regime history ----------------------------------------------------

    def save_regime(self, regime: RegimeState) -> None:
        with self._session() as s:
            s.add(RegimeHistoryRow(
                date=regime.timestamp,
                level=regime.level.value,
                allocation_pct=regime.allocation_pct,
                breadth=regime.breadth,
                bench_close=regime.bench_close,
                bench_50dma=regime.bench_50dma,
                bench_200dma=regime.bench_200dma,
                bench_3m_return=regime.bench_3m_return,
            ))
            s.commit()

    # -- Rebalance history -------------------------------------------------

    def save_rebalance(
        self,
        rebalance_date: datetime,
        num_selected: int,
        allocation_pct: float,
        picks: dict[str, float],
    ) -> None:
        with self._session() as s:
            s.add(RebalanceHistoryRow(
                rebalance_date=rebalance_date,
                num_selected=num_selected,
                allocation_pct=allocation_pct,
                picks_json=json.dumps(picks),
            ))
            s.commit()

    def has_rebalanced_today(self) -> bool:
        """Check if a rebalance has already been recorded for today."""
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = datetime.combine(date.today(), datetime.max.time())
        with self._session() as s:
            count = (
                s.query(RebalanceHistoryRow)
                .filter(
                    RebalanceHistoryRow.rebalance_date >= today_start,
                    RebalanceHistoryRow.rebalance_date <= today_end,
                )
                .count()
            )
            return count > 0

    # -- Stopped-out tracker -----------------------------------------------

    def save_stopped_out(self, ticker: str, weight: float, stop_date: datetime) -> None:
        month_year = stop_date.strftime("%Y-%m")
        with self._session() as s:
            s.add(StoppedOutRow(
                ticker=ticker,
                weight=weight,
                stop_date=stop_date,
                month_year=month_year,
            ))
            s.commit()

    def load_stopped_out_this_month(self) -> dict[str, float]:
        month_year = datetime.now().strftime("%Y-%m")
        with self._session() as s:
            rows = (
                s.query(StoppedOutRow)
                .filter(StoppedOutRow.month_year == month_year)
                .all()
            )
            return {r.ticker: r.weight for r in rows}

    def clear_stopped_out_month(self, month_year: str) -> None:
        with self._session() as s:
            s.query(StoppedOutRow).filter(StoppedOutRow.month_year == month_year).delete()
            s.commit()
