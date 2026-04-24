"""Tests for signal_generator: assess_regime, score_and_rank, generate_rebalance_signals."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from constants import SECTOR_MAP
from models import OrderAction, Position, RegimeLevel
from signal_generator import assess_regime, generate_rebalance_signals, score_and_rank
from utils import adv_126, rolling_volatility, sma


# ---------------------------------------------------------------------------
# assess_regime
# ---------------------------------------------------------------------------

class TestAssessRegime:
    def _uptrending_bench(self, n=300) -> pd.Series:
        """Benchmark in a clear uptrend: above 200DMA, golden cross, positive 3M."""
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        prices = 10000 * np.cumprod(1 + np.full(n, 0.0005))
        return pd.Series(prices, index=dates)

    def _downtrending_bench(self, n=300) -> pd.Series:
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        prices = 10000 * np.cumprod(1 + np.full(n, -0.001))
        return pd.Series(prices, index=dates)

    def _make_stocks(self, n=300, count=20, trend=0.0005):
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        data = {}
        for i in range(count):
            data[f"STOCK{i}"] = 500 * np.cumprod(1 + np.full(n, trend) + np.random.default_rng(i).normal(0, 0.005, n))
        return pd.DataFrame(data, index=dates)

    def test_full_risk_on(self, sample_config):
        bench = self._uptrending_bench()
        stocks = self._make_stocks(trend=0.001)
        regime = assess_regime(bench, stocks, sample_config)
        assert regime.level == RegimeLevel.FULL_RISK_ON
        assert regime.allocation_pct == 1.0

    def test_risk_off_in_downtrend(self, sample_config):
        bench = self._downtrending_bench()
        stocks = self._make_stocks(trend=-0.002)
        regime = assess_regime(bench, stocks, sample_config)
        assert regime.level == RegimeLevel.RISK_OFF
        assert regime.allocation_pct == 0.0

    def test_fields_populated(self, sample_config):
        bench = self._uptrending_bench()
        stocks = self._make_stocks()
        regime = assess_regime(bench, stocks, sample_config)
        assert regime.bench_close > 0
        assert regime.bench_50dma > 0
        assert regime.bench_200dma > 0
        assert 0 <= regime.breadth <= 1


# ---------------------------------------------------------------------------
# score_and_rank
# ---------------------------------------------------------------------------

class TestScoreAndRank:
    def test_returns_non_empty_for_valid_data(self, price_data, sample_config):
        close = price_data["close"].drop(columns=["NIFTY 200"])
        volume = price_data["volume"].drop(columns=["NIFTY 200"])
        daily_ret = close.pct_change().fillna(0)
        vol_60 = rolling_volatility(daily_ret, 60)
        adv = adv_126(close, volume)
        dma_100 = sma(close, 100)
        dma_200 = sma(close, 200)

        result = score_and_rank(
            close, volume, vol_60, adv, dma_100, dma_200, SECTOR_MAP, sample_config,
        )
        assert not result.empty
        assert "risk_adj_score" in result.columns
        assert "score" in result.columns
        assert "vol" in result.columns

    def test_sorted_descending_by_score(self, price_data, sample_config):
        close = price_data["close"].drop(columns=["NIFTY 200"])
        volume = price_data["volume"].drop(columns=["NIFTY 200"])
        daily_ret = close.pct_change().fillna(0)
        vol_60 = rolling_volatility(daily_ret, 60)
        adv = adv_126(close, volume)
        dma_100 = sma(close, 100)
        dma_200 = sma(close, 200)

        result = score_and_rank(
            close, volume, vol_60, adv, dma_100, dma_200, SECTOR_MAP, sample_config,
        )
        if len(result) > 1:
            scores = result["risk_adj_score"].values
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ---------------------------------------------------------------------------
# generate_rebalance_signals
# ---------------------------------------------------------------------------

class TestGenerateRebalanceSignals:
    def test_sell_signal_for_dropped_position(self, price_data, sample_config):
        close = price_data["close"].drop(columns=["NIFTY 200"])
        high = price_data["high"].drop(columns=["NIFTY 200"])
        low = price_data["low"].drop(columns=["NIFTY 200"])
        volume = price_data["volume"].drop(columns=["NIFTY 200"])
        bench = price_data["close"]["NIFTY 200"]

        fake_pos = {
            "NONEXISTENT_STOCK": Position(
                ticker="NONEXISTENT_STOCK", weight=0.10, entry_price=100,
                entry_date=datetime.now() - timedelta(days=30),
                high_watermark=110, sector="Other",
            ),
        }
        result = generate_rebalance_signals(
            close, high, low, volume, bench, fake_pos, sample_config,
        )
        sell_tickers = {s.ticker for s in result.sells}
        assert "NONEXISTENT_STOCK" in sell_tickers

    def test_no_signals_in_risk_off(self, price_data, sample_config):
        sample_config["regime"]["breadth_threshold"] = 0.99
        sample_config["regime"]["neutral_breadth_threshold"] = 0.99

        close = price_data["close"].drop(columns=["NIFTY 200"])
        high = price_data["high"].drop(columns=["NIFTY 200"])
        low = price_data["low"].drop(columns=["NIFTY 200"])
        volume = price_data["volume"].drop(columns=["NIFTY 200"])

        n = len(close)
        bench = pd.Series(
            10000 * np.cumprod(1 + np.full(n, -0.002)),
            index=close.index,
        )

        result = generate_rebalance_signals(
            close, high, low, volume, bench, {}, sample_config,
        )
        assert result.regime.level == RegimeLevel.RISK_OFF
        assert len(result.buys) == 0

    def test_result_structure(self, price_data, sample_config):
        close = price_data["close"].drop(columns=["NIFTY 200"])
        high = price_data["high"].drop(columns=["NIFTY 200"])
        low = price_data["low"].drop(columns=["NIFTY 200"])
        volume = price_data["volume"].drop(columns=["NIFTY 200"])
        bench = price_data["close"]["NIFTY 200"]

        result = generate_rebalance_signals(
            close, high, low, volume, bench, {}, sample_config,
        )
        assert hasattr(result, "buys")
        assert hasattr(result, "sells")
        assert hasattr(result, "regime")
        assert hasattr(result, "target_weights")
        for sig in result.buys:
            assert sig.action == OrderAction.BUY
        for sig in result.sells:
            assert sig.action in (OrderAction.SELL, OrderAction.REBAL_EXIT)
