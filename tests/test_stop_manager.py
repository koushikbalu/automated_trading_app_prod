"""Tests for stop_manager: initial stops, trailing stops, batch checks, re-entry."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from models import OrderAction, Position
from stop_manager import (
    check_all_stops,
    check_re_entry,
    check_stop_triggered,
    compute_initial_stop,
    update_trailing_stop,
)


# ---------------------------------------------------------------------------
# compute_initial_stop
# ---------------------------------------------------------------------------

class TestComputeInitialStop:
    def test_basic(self):
        stop = compute_initial_stop(1000, 50, 2.0)
        assert stop == pytest.approx(900.0)

    def test_zero_atr_returns_none(self):
        assert compute_initial_stop(1000, 0, 2.0) is None

    def test_nan_atr_returns_none(self):
        assert compute_initial_stop(1000, float("nan"), 2.0) is None

    def test_negative_atr_returns_none(self):
        assert compute_initial_stop(1000, -10, 2.0) is None


# ---------------------------------------------------------------------------
# update_trailing_stop
# ---------------------------------------------------------------------------

class TestUpdateTrailingStop:
    def _pos(self, entry=1000, hwm=1050, stop=900):
        return Position(
            ticker="TEST", weight=0.10, entry_price=entry,
            entry_date=datetime.now(), high_watermark=hwm,
            stop_price=stop, sector="IT",
        )

    def test_ratchets_stop_up(self):
        pos = self._pos(entry=1000, hwm=1000, stop=900)
        update_trailing_stop(pos, 1100, atr_value=50, atr_multiple=2.0)
        assert pos.high_watermark == 1100
        assert pos.stop_price == pytest.approx(1000.0)

    def test_never_lowers_stop(self):
        pos = self._pos(entry=1000, hwm=1100, stop=1000)
        update_trailing_stop(pos, 1050, atr_value=50, atr_multiple=2.0)
        assert pos.stop_price == 1000

    def test_sets_stop_when_none(self):
        pos = self._pos(entry=1000, hwm=1000, stop=None)
        update_trailing_stop(pos, 1050, atr_value=50, atr_multiple=2.0)
        assert pos.stop_price is not None
        assert pos.stop_price == pytest.approx(950.0)


# ---------------------------------------------------------------------------
# check_stop_triggered
# ---------------------------------------------------------------------------

class TestCheckStopTriggered:
    def _pos(self, stop_price):
        return Position(
            ticker="TEST", weight=0.10, entry_price=1000,
            entry_date=datetime.now(), high_watermark=1100,
            stop_price=stop_price, sector="IT",
        )

    def test_triggered_below_stop(self):
        triggered, reason = check_stop_triggered(self._pos(950), 940)
        assert triggered is True
        assert "ATR" in reason

    def test_not_triggered_above_stop(self):
        triggered, _ = check_stop_triggered(self._pos(950), 960)
        assert triggered is False

    def test_no_stop_price(self):
        triggered, _ = check_stop_triggered(self._pos(None), 500)
        assert triggered is False


# ---------------------------------------------------------------------------
# check_all_stops (batch)
# ---------------------------------------------------------------------------

class TestCheckAllStops:
    def test_stop_triggered_for_one(self, sample_config):
        positions = {
            "RELIANCE": Position(
                ticker="RELIANCE", weight=0.12, entry_price=2500,
                entry_date=datetime.now() - timedelta(days=10),
                high_watermark=2700, stop_price=2400, sector="Energy",
            ),
            "TCS": Position(
                ticker="TCS", weight=0.10, entry_price=3400,
                entry_date=datetime.now() - timedelta(days=10),
                high_watermark=3600, stop_price=3200, sector="IT",
            ),
        }
        live = {"RELIANCE": 2300, "TCS": 3500}
        atr = {"RELIANCE": 80, "TCS": 100}

        exits, updated = check_all_stops(positions, live, atr, sample_config)
        exit_tickers = {s.ticker for s in exits}
        assert "RELIANCE" in exit_tickers
        assert "TCS" not in exit_tickers

    def test_hard_stop_triggers(self, sample_config):
        pos = {
            "INFY": Position(
                ticker="INFY", weight=0.10, entry_price=1500,
                entry_date=datetime.now() - timedelta(days=5),
                high_watermark=1500, stop_price=None, sector="IT",
            ),
        }
        live = {"INFY": 1370}  # -8.67% from entry, below -8% hard stop
        atr = {}  # no ATR so trailing stop won't trigger

        exits, _ = check_all_stops(pos, live, atr, sample_config)
        assert len(exits) == 1
        assert "Hard stop" in exits[0].reason

    def test_no_exits_when_above_stops(self, sample_config):
        pos = {
            "TCS": Position(
                ticker="TCS", weight=0.10, entry_price=3400,
                entry_date=datetime.now(), high_watermark=3600,
                stop_price=3200, sector="IT",
            ),
        }
        exits, _ = check_all_stops(pos, {"TCS": 3500}, {"TCS": 100}, sample_config)
        assert len(exits) == 0

    def test_missing_live_price_skipped(self, sample_config):
        pos = {
            "TCS": Position(
                ticker="TCS", weight=0.10, entry_price=3400,
                entry_date=datetime.now(), high_watermark=3600,
                stop_price=3200, sector="IT",
            ),
        }
        exits, _ = check_all_stops(pos, {}, {"TCS": 100}, sample_config)
        assert len(exits) == 0


# ---------------------------------------------------------------------------
# check_re_entry
# ---------------------------------------------------------------------------

class TestCheckReEntry:
    def test_re_entry_when_above_20dma_and_top_ranked(self, sample_config):
        stopped_out = {"RELIANCE": 0.10}
        ranked = pd.DataFrame(
            {"risk_adj_score": [5, 4, 3]},
            index=["RELIANCE", "TCS", "INFY"],
        )
        live = {"RELIANCE": 2600}
        dma_20 = {"RELIANCE": 2500}
        atr = {"RELIANCE": 80}

        signals = check_re_entry(
            stopped_out, ranked, live, dma_20, atr,
            current_positions={}, top_n=10, cfg=sample_config,
        )
        assert len(signals) == 1
        assert signals[0].ticker == "RELIANCE"
        assert signals[0].action == OrderAction.BUY

    def test_no_re_entry_below_20dma(self, sample_config):
        stopped_out = {"RELIANCE": 0.10}
        ranked = pd.DataFrame({"risk_adj_score": [5]}, index=["RELIANCE"])
        live = {"RELIANCE": 2400}
        dma_20 = {"RELIANCE": 2500}
        atr = {"RELIANCE": 80}

        signals = check_re_entry(
            stopped_out, ranked, live, dma_20, atr,
            current_positions={}, top_n=10, cfg=sample_config,
        )
        assert len(signals) == 0

    def test_no_re_entry_if_not_top_ranked(self, sample_config):
        stopped_out = {"RELIANCE": 0.10}
        ranked = pd.DataFrame({"risk_adj_score": [5]}, index=["TCS"])
        live = {"RELIANCE": 2600}
        dma_20 = {"RELIANCE": 2500}

        signals = check_re_entry(
            stopped_out, ranked, live, dma_20, {},
            current_positions={}, top_n=1, cfg=sample_config,
        )
        assert len(signals) == 0

    def test_no_re_entry_when_disabled(self, sample_config):
        sample_config["risk"]["re_entry_enabled"] = False
        stopped_out = {"RELIANCE": 0.10}
        ranked = pd.DataFrame({"risk_adj_score": [5]}, index=["RELIANCE"])

        signals = check_re_entry(
            stopped_out, ranked, {"RELIANCE": 2600}, {"RELIANCE": 2500}, {},
            current_positions={}, top_n=10, cfg=sample_config,
        )
        assert len(signals) == 0

    def test_skip_already_held(self, sample_config):
        stopped_out = {"RELIANCE": 0.10}
        ranked = pd.DataFrame({"risk_adj_score": [5]}, index=["RELIANCE"])
        current = {
            "RELIANCE": Position(
                ticker="RELIANCE", weight=0.10, entry_price=2500,
                entry_date=datetime.now(), high_watermark=2500,
                sector="Energy",
            ),
        }

        signals = check_re_entry(
            stopped_out, ranked, {"RELIANCE": 2600}, {"RELIANCE": 2500}, {},
            current_positions=current, top_n=10, cfg=sample_config,
        )
        assert len(signals) == 0
