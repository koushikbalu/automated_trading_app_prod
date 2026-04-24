"""Tests for state_manager: CRUD round-trips using SQLite in-memory."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from models import Position, PortfolioSnapshot, RegimeLevel, RegimeState, TradeRecord
from state_manager import StateManager


@pytest.fixture()
def sm() -> StateManager:
    return StateManager(db_url="sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestPositions:
    def test_save_and_load(self, sm):
        pos = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            stop_price=3200, sector="IT",
        )
        sm.save_position(pos)
        loaded = sm.load_positions()
        assert "TCS" in loaded
        assert loaded["TCS"].weight == pytest.approx(0.10)
        assert loaded["TCS"].sector == "IT"

    def test_update_existing(self, sm):
        pos = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            stop_price=3200, sector="IT",
        )
        sm.save_position(pos)
        pos.weight = 0.15
        sm.save_position(pos)
        loaded = sm.load_positions()
        assert loaded["TCS"].weight == pytest.approx(0.15)

    def test_close_position(self, sm):
        pos = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            sector="IT",
        )
        sm.save_position(pos)
        sm.close_position("TCS")
        loaded = sm.load_positions()
        assert "TCS" not in loaded

    def test_update_prices(self, sm):
        pos = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            sector="IT",
        )
        sm.save_position(pos)
        sm.update_position_prices({"TCS": 3500})


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

class TestTrades:
    def test_record_and_query(self, sm):
        trade = TradeRecord(
            date=datetime.now(), ticker="TCS", action="BUY",
            price=3400, weight_traded=0.10, reason="test",
            order_id="ORD001", costs=0.001,
        )
        sm.record_trade(trade)
        recent = sm.get_recent_trades(10)
        assert len(recent) == 1
        assert recent[0]["ticker"] == "TCS"

    def test_order_by_date_desc(self, sm):
        for i in range(5):
            sm.record_trade(TradeRecord(
                date=datetime.now() - timedelta(days=5 - i),
                ticker=f"STOCK{i}", action="BUY",
                price=100, weight_traded=0.05, reason="test",
            ))
        recent = sm.get_recent_trades(5)
        dates = [r["date"] for r in recent]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

class TestPortfolioState:
    def test_save_and_load_latest(self, sm):
        snap = PortfolioSnapshot(
            date=datetime.now(), portfolio_value=20_000_000,
            portfolio_peak=21_000_000, circuit_breaker_active=False,
            cash_weight=0.70, exposure=0.30, positions_count=3,
        )
        sm.save_portfolio_state(snap)
        loaded = sm.load_latest_portfolio_state()
        assert loaded is not None
        assert loaded.portfolio_value == pytest.approx(20_000_000)

    def test_load_returns_none_when_empty(self, sm):
        assert sm.load_latest_portfolio_state() is None

    def test_load_snapshot_for_date(self, sm):
        d = datetime.now().date()
        snap = PortfolioSnapshot(
            date=datetime.now(), portfolio_value=20_000_000,
            portfolio_peak=21_000_000, circuit_breaker_active=False,
            cash_weight=0.70, exposure=0.30, positions_count=3,
        )
        sm.save_portfolio_state(snap)
        loaded = sm.load_snapshot_for_date(d)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Regime history
# ---------------------------------------------------------------------------

class TestRegimeHistory:
    def test_save_regime(self, sm):
        regime = RegimeState(
            level=RegimeLevel.FULL_RISK_ON,
            allocation_pct=1.0,
            breadth=0.55,
            bench_close=15000,
            bench_50dma=14500,
            bench_200dma=13500,
            bench_3m_return=0.05,
        )
        sm.save_regime(regime)


# ---------------------------------------------------------------------------
# Rebalance history
# ---------------------------------------------------------------------------

class TestRebalanceHistory:
    def test_save_and_check_today(self, sm):
        assert sm.has_rebalanced_today() is False
        sm.save_rebalance(datetime.now(), 10, 1.0, {"TCS": 0.10})
        assert sm.has_rebalanced_today() is True


# ---------------------------------------------------------------------------
# Stopped-out tracker
# ---------------------------------------------------------------------------

class TestStoppedOut:
    def test_save_and_load_this_month(self, sm):
        sm.save_stopped_out("TCS", 0.10, datetime.now())
        loaded = sm.load_stopped_out_this_month()
        assert "TCS" in loaded

    def test_clear_month(self, sm):
        sm.save_stopped_out("TCS", 0.10, datetime.now())
        month_year = datetime.now().strftime("%Y-%m")
        sm.clear_stopped_out_month(month_year)
        loaded = sm.load_stopped_out_this_month()
        assert len(loaded) == 0


# ---------------------------------------------------------------------------
# Trade fill status update (webhook)
# ---------------------------------------------------------------------------

class TestTradeWebhook:
    def test_update_fill_status(self, sm):
        trade = TradeRecord(
            date=datetime.now(), ticker="TCS", action="BUY",
            price=3400, weight_traded=0.10, reason="test",
            order_id="ORD999", fill_status="PENDING",
        )
        sm.record_trade(trade)
        sm.update_trade_fill_status("ORD999", "COMPLETE", 100)

    def test_update_missing_order_no_crash(self, sm):
        sm.update_trade_fill_status("NONEXISTENT", "COMPLETE", 50)
