"""Tests for risk_manager: CircuitBreaker, enforce_sector_caps, validate_order."""

from __future__ import annotations

import pandas as pd
import pytest

from models import Position, RegimeLevel
from risk_manager import CircuitBreaker, enforce_sector_caps, validate_order


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def _regime(self, level: RegimeLevel):
        return type("R", (), {"level": level})()

    def test_triggers_on_drawdown(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=5)
        triggered = cb.check(79_000, 100_000, self._regime(RegimeLevel.FULL_RISK_ON))
        assert triggered is True
        assert cb.active is True

    def test_no_trigger_within_threshold(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=5)
        triggered = cb.check(85_000, 100_000, self._regime(RegimeLevel.FULL_RISK_ON))
        assert triggered is False
        assert cb.active is False

    def test_resets_after_consecutive_risk_on_days(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=3)
        cb.active = True
        regime_on = self._regime(RegimeLevel.FULL_RISK_ON)

        assert cb.check(90_000, 100_000, regime_on) is True  # day 1
        assert cb.check(90_000, 100_000, regime_on) is True  # day 2
        assert cb.check(90_000, 100_000, regime_on) is False  # day 3: reset
        assert cb.active is False

    def test_reset_streak_broken_by_non_risk_on(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=3)
        cb.active = True
        regime_on = self._regime(RegimeLevel.FULL_RISK_ON)
        regime_off = self._regime(RegimeLevel.RISK_OFF)

        cb.check(90_000, 100_000, regime_on)   # streak=1
        cb.check(90_000, 100_000, regime_off)  # streak=0
        cb.check(90_000, 100_000, regime_on)   # streak=1
        assert cb.active is True

    def test_instant_reset_with_1_day(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=1)
        cb.active = True
        result = cb.check(90_000, 100_000, self._regime(RegimeLevel.FULL_RISK_ON))
        assert result is False
        assert cb.active is False

    def test_zero_peak_no_crash(self):
        cb = CircuitBreaker(threshold=-0.20, reset_days=5)
        triggered = cb.check(0, 0, self._regime(RegimeLevel.FULL_RISK_ON))
        assert triggered is False


# ---------------------------------------------------------------------------
# enforce_sector_caps
# ---------------------------------------------------------------------------

class TestEnforceSectorCaps:
    def test_no_change_when_under_cap(self):
        weights = {"A": 0.10, "B": 0.10, "C": 0.10}
        sector_map = {"A": "IT", "B": "Banking", "C": "Energy"}
        result = enforce_sector_caps(weights, 0.30, sector_map)
        assert sum(result.values()) == pytest.approx(0.30, abs=1e-6)

    def test_scales_down_over_cap_sector(self):
        weights = {"A": 0.15, "B": 0.15, "C": 0.10, "D": 0.10, "E": 0.10}
        sector_map = {"A": "IT", "B": "IT", "C": "Energy", "D": "Banking", "E": "Pharma"}
        original_total = sum(weights.values())
        result = enforce_sector_caps(weights, 0.25, sector_map)
        it_weight = result["A"] + result["B"]
        assert it_weight < 0.30  # IT was scaled down from 0.30
        assert sum(result.values()) == pytest.approx(original_total, abs=1e-6)

    def test_preserves_total_weight(self):
        weights = {"A": 0.20, "B": 0.15, "C": 0.05, "D": 0.10}
        sector_map = {"A": "IT", "B": "IT", "C": "Energy", "D": "Banking"}
        original_total = sum(weights.values())
        result = enforce_sector_caps(weights, 0.25, sector_map)
        assert sum(result.values()) == pytest.approx(original_total, abs=1e-6)

    def test_empty_weights(self):
        result = enforce_sector_caps({}, 0.30)
        assert result == {}


# ---------------------------------------------------------------------------
# validate_order
# ---------------------------------------------------------------------------

class TestValidateOrder:
    def test_valid_order(self):
        ok, reason = validate_order("TCS", 0.10, 20_000_000, 5_000_000, 5, 10)
        assert ok is True
        assert reason == ""

    def test_insufficient_capital(self):
        ok, reason = validate_order("TCS", 0.10, 20_000_000, 1_000_000, 5, 10)
        assert ok is False
        assert "Insufficient capital" in reason

    def test_max_positions_reached(self):
        ok, reason = validate_order("TCS", 0.10, 20_000_000, 5_000_000, 10, 10)
        assert ok is False
        assert "Max positions" in reason

    def test_order_value_too_small(self):
        ok, reason = validate_order("TCS", 0.00001, 20_000_000, 5_000_000, 5, 10)
        assert ok is False
        assert "too small" in reason

    def test_freeze_qty_exceeded(self):
        ok, reason = validate_order(
            "RELIANCE", 0.50, 20_000_000, 15_000_000, 1, 10, price=2500.0
        )
        assert ok is False
        assert "freeze limit" in reason
