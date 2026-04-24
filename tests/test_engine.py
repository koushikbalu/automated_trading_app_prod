"""Integration tests for TradingEngine with mocked broker dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from models import OrderAction, Position, PortfolioSnapshot, RegimeLevel


@pytest.fixture()
def engine_cfg(sample_config):
    sample_config["persistence"]["database_url"] = "sqlite:///:memory:"
    sample_config["broker"]["dry_run"] = True
    return sample_config


@pytest.fixture()
def engine(engine_cfg):
    with patch("engine.TelegramNotifier") as mock_notifier:
        mock_notifier.return_value = MagicMock()
        from engine import TradingEngine
        eng = TradingEngine(engine_cfg)
        return eng


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_default_values(self, engine):
        assert engine.portfolio_value > 0
        assert engine.portfolio_peak >= engine.portfolio_value
        assert engine.circuit_breaker.active is False
        assert isinstance(engine.positions, dict)

    def test_loads_config(self, engine, engine_cfg):
        assert engine.cfg is engine_cfg


# ---------------------------------------------------------------------------
# get_status / get_stops / get_health
# ---------------------------------------------------------------------------

class TestStatusEndpoints:
    def test_get_status_empty(self, engine):
        st = engine.get_status()
        assert st["positions_count"] == 0
        assert st["exposure"] == pytest.approx(0.0)

    def test_get_status_with_positions(self, engine):
        engine.positions["TCS"] = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            stop_price=3200, sector="IT",
        )
        st = engine.get_status()
        assert st["positions_count"] == 1
        assert st["exposure"] == pytest.approx(0.10)

    def test_get_stops_empty(self, engine):
        assert engine.get_stops() == []

    def test_get_stops_with_positions(self, engine):
        engine.positions["TCS"] = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            stop_price=3200, sector="IT",
        )
        stops = engine.get_stops()
        assert len(stops) == 1
        assert stops[0]["ticker"] == "TCS"

    def test_get_health(self, engine):
        h = engine.get_health()
        assert "db_connected" in h
        assert "token_valid" in h
        assert "portfolio_value" in h


# ---------------------------------------------------------------------------
# _save_snapshot / _current_snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_save_and_restore(self, engine):
        engine.portfolio_value = 21_000_000
        engine.portfolio_peak = 21_500_000
        engine._save_snapshot()

        snap = engine.state.load_latest_portfolio_state()
        assert snap is not None
        assert snap.portfolio_value == pytest.approx(21_000_000)
        assert snap.portfolio_peak == pytest.approx(21_500_000)


# ---------------------------------------------------------------------------
# _validate_signals
# ---------------------------------------------------------------------------

class TestValidateSignals:
    def test_sell_always_passes(self, engine):
        from models import Signal
        sig = Signal("TCS", OrderAction.SELL, 0.0, 0.10, 3400, "stop")
        result = engine._validate_signals([sig])
        assert len(result) == 1

    def test_buy_rejected_when_over_position_limit(self, engine):
        from models import Signal
        for i in range(10):
            engine.positions[f"STOCK{i}"] = Position(
                ticker=f"STOCK{i}", weight=0.09, entry_price=1000,
                entry_date=datetime.now(), high_watermark=1000, sector="IT",
            )
        engine.available_cash = 5_000_000

        sig = Signal("NEWSTOCK", OrderAction.BUY, 0.10, 0.0, 1000, "buy")
        result = engine._validate_signals([sig])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# monitor_stops (dry-run, no live data)
# ---------------------------------------------------------------------------

class TestMonitorStops:
    @patch("engine.get_live_quotes", return_value={})
    @patch("engine.fetch_historical")
    def test_no_crash_with_empty_prices(self, mock_fetch, mock_quotes, engine):
        engine.positions["TCS"] = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now(), high_watermark=3600,
            stop_price=3200, sector="IT",
        )
        mock_fetch.return_value = pd.DataFrame()
        engine._monitor_stops_impl()

    @patch("engine.get_live_quotes")
    @patch("engine.fetch_historical")
    def test_stop_triggered_removes_position(self, mock_fetch, mock_quotes, engine):
        engine.positions["TCS"] = Position(
            ticker="TCS", weight=0.10, entry_price=3400,
            entry_date=datetime.now() - timedelta(days=10),
            high_watermark=3600, stop_price=3200, sector="IT",
        )
        mock_quotes.return_value = {"TCS": 3100}
        mock_fetch.return_value = pd.DataFrame({
            "close": [3400 + i * 5 for i in range(20)],
            "high": [3410 + i * 5 for i in range(20)],
            "low": [3390 + i * 5 for i in range(20)],
        })
        engine._monitor_stops_impl()
        assert "TCS" not in engine.positions
