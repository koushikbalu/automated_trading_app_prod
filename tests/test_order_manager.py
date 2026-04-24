"""Tests for order_manager: qty calc, dry-run, dedup, sliced orders."""

from __future__ import annotations

from datetime import datetime

import pytest

from models import OrderAction, Signal
from order_manager import OrderManager


@pytest.fixture()
def om(sample_config) -> OrderManager:
    return OrderManager(sample_config)


# ---------------------------------------------------------------------------
# _weight_to_qty
# ---------------------------------------------------------------------------

class TestWeightToQty:
    def test_basic_calculation(self, om):
        qty = om._weight_to_qty(0.10, 2000.0, 20_000_000)
        assert qty == 1000

    def test_zero_price(self, om):
        assert om._weight_to_qty(0.10, 0, 20_000_000) == 0

    def test_zero_weight(self, om):
        assert om._weight_to_qty(0, 2000.0, 20_000_000) == 0

    def test_negative_weight(self, om):
        assert om._weight_to_qty(-0.05, 2000.0, 20_000_000) == 0

    def test_fractional_truncation(self, om):
        qty = om._weight_to_qty(0.10, 3000.0, 20_000_000)
        assert qty == 666  # 2_000_000 / 3000 = 666.66 -> 666


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_returns_trade_record(self, om):
        sig = Signal(
            ticker="RELIANCE", action=OrderAction.BUY,
            target_weight=0.10, current_weight=0.0,
            price=2500.0, reason="test",
        )
        tr = om.place_order(sig, 20_000_000)
        assert tr.fill_status == "DRY_RUN"
        assert tr.ticker == "RELIANCE"
        assert tr.action == "BUY"
        assert tr.weight_traded > 0

    def test_dry_run_sell(self, om):
        sig = Signal(
            ticker="TCS", action=OrderAction.SELL,
            target_weight=0.0, current_weight=0.10,
            price=3400.0, reason="stop",
        )
        tr = om.place_order(sig, 20_000_000)
        assert tr.fill_status == "DRY_RUN"
        assert tr.action == "SELL"


# ---------------------------------------------------------------------------
# Duplicate blocking
# ---------------------------------------------------------------------------

class TestDuplicateBlocking:
    def test_second_order_same_ticker_blocked(self, om):
        sig = Signal(
            ticker="TCS", action=OrderAction.BUY,
            target_weight=0.10, current_weight=0.0,
            price=3400.0, reason="test",
        )
        tr1 = om.place_order(sig, 20_000_000)
        tr2 = om.place_order(sig, 20_000_000)
        assert tr1.fill_status == "DRY_RUN"
        assert tr2.fill_status == "SKIPPED"
        assert tr2.reason == "DUPLICATE_BLOCKED"


# ---------------------------------------------------------------------------
# place_orders (sells-first ordering)
# ---------------------------------------------------------------------------

class TestPlaceOrders:
    def test_sells_before_buys(self, om):
        signals = [
            Signal("BUY1", OrderAction.BUY, 0.10, 0.0, 1000, "buy"),
            Signal("SELL1", OrderAction.SELL, 0.0, 0.10, 1000, "sell"),
            Signal("BUY2", OrderAction.BUY, 0.10, 0.0, 2000, "buy"),
            Signal("SELL2", OrderAction.REBAL_EXIT, 0.0, 0.08, 1500, "exit"),
        ]
        records = om.place_orders(signals, 20_000_000)
        actions = [r.action for r in records]
        sell_indices = [i for i, a in enumerate(actions) if a == "SELL"]
        buy_indices = [i for i, a in enumerate(actions) if a == "BUY"]
        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices)


# ---------------------------------------------------------------------------
# Sliced orders
# ---------------------------------------------------------------------------

class TestSlicedOrders:
    def test_large_order_gets_sliced(self, sample_config):
        sample_config["execution"]["max_slice_value"] = 500_000
        sample_config["execution"]["slice_delay_seconds"] = 0
        om = OrderManager(sample_config)

        sig = Signal(
            ticker="RELIANCE", action=OrderAction.BUY,
            target_weight=0.10, current_weight=0.0,
            price=2500.0, reason="test",
        )
        records = om.place_order_sliced(sig, 20_000_000)
        assert len(records) > 1

    def test_small_order_not_sliced(self, om):
        sig = Signal(
            ticker="RELIANCE", action=OrderAction.BUY,
            target_weight=0.01, current_weight=0.0,
            price=2500.0, reason="test",
        )
        records = om.place_order_sliced(sig, 20_000_000)
        assert len(records) == 1
