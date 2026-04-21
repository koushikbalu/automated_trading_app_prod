"""TradingEngine -- the central orchestrator that connects all modules.

Scheduled jobs call these methods:
- monitor_stops()         every 5 min (09:15-15:30)
- check_re_entry()        at 15:20 daily
- daily_rebalance()       at 15:45 on last trading day of month
- daily_summary()         at 16:00 daily
- refresh_token()         at 09:10 daily
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from constants import BROAD_UNIVERSE, SECTOR_MAP
from data_manager import fetch_historical, fetch_historical_bulk, get_kite, get_live_quotes
from models import (
    OrderAction,
    Position,
    PortfolioSnapshot,
    RegimeLevel,
    Signal,
)
from notifier import TelegramNotifier
from nse_calendar import is_last_trading_day as _nse_is_last_trading_day, is_trading_day
from order_manager import OrderManager
from risk_manager import CircuitBreaker, apply_exposure_floor, enforce_sector_caps
from signal_generator import assess_regime, generate_rebalance_signals, score_and_rank
from state_manager import StateManager
from stop_manager import check_all_stops, check_re_entry, compute_initial_stop
from token_manager import TokenManager
from utils import compute_atr_df, compute_atr_series, rolling_volatility, sma, adv_126

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _is_last_trading_day() -> bool:
    """Check if today is the last NSE trading day of the month,
    accounting for weekends *and* market holidays."""
    return _nse_is_last_trading_day(datetime.now().date())


class TradingEngine:
    """Central orchestrator connecting data, strategy, risk, execution,
    state, and notifications."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or _load_config()
        self.state = StateManager()
        self.notifier = TelegramNotifier()
        self.order_mgr = OrderManager(self.cfg, notifier=self.notifier)
        self.token_mgr = TokenManager()
        risk_cfg = self.cfg.get("risk", {})
        self.circuit_breaker = CircuitBreaker(
            threshold=risk_cfg.get("drawdown_circuit_breaker", -0.20),
            reset_days=risk_cfg.get("cb_reset_days", 5),
        )

        cap_cfg = self.cfg.get("capital", {})
        self.initial_capital = cap_cfg.get("initial", 20_000_000)

        snap = self.state.load_latest_portfolio_state()
        if snap:
            self.portfolio_value = snap.portfolio_value
            self.portfolio_peak = snap.portfolio_peak
            self.circuit_breaker.active = snap.circuit_breaker_active
        else:
            self.portfolio_value = self.initial_capital
            self.portfolio_peak = self.initial_capital

        self.positions = self.state.load_positions()
        self.stopped_out = self.state.load_stopped_out_this_month()
        self._last_ranked: pd.DataFrame = pd.DataFrame()
        self._last_ranked_date: datetime | None = None
        self.available_cash: float = self.portfolio_value - sum(
            p.weight * self.portfolio_value for p in self.positions.values()
        )
        self._token_valid: bool = True
        self._last_stop_check: datetime | None = None
        self._last_rebalance: datetime | None = None
        self._atr_cache: dict[str, float] = {}
        self._atr_cache_date: datetime | None = None
        self._regime_cache: tuple[datetime | None, object] = (None, None)

    # ------------------------------------------------------------------
    # Portfolio value sync from broker
    # ------------------------------------------------------------------

    def _sync_portfolio_value(self) -> None:
        """Sync portfolio value from live broker data.

        In dry-run mode, falls back to snapshot-based calculation.
        """
        if self.order_mgr.dry_run:
            return

        try:
            kite = get_kite()
            margins = kite.margins("equity")
            available_cash = float(margins.get("available", {}).get("cash", 0))

            live_prices = get_live_quotes(list(self.positions.keys()))
            invested_value = 0.0
            for ticker, pos in self.positions.items():
                price = live_prices.get(ticker, pos.entry_price)
                invested_value += pos.weight * self.portfolio_value * (price / pos.entry_price)

            self.portfolio_value = available_cash + invested_value
            self.portfolio_peak = max(self.portfolio_peak, self.portfolio_value)
            self.available_cash = available_cash
            logger.info(
                "Portfolio synced: value=%.2f, cash=%.2f, invested=%.2f",
                self.portfolio_value, available_cash, invested_value,
            )
        except Exception as exc:
            logger.error("Portfolio sync failed, using last known value: %s", exc)

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def _reconcile_positions(self) -> None:
        """Compare internal position state against broker holdings.

        Logs warnings for any mismatches. Sends Telegram alerts for
        critical divergences (>5% weight difference). Does NOT auto-correct.
        """
        if self.order_mgr.dry_run:
            return

        try:
            kite = get_kite()
            broker_positions = kite.positions().get("net", [])
        except Exception as exc:
            logger.error("Position reconciliation failed: %s", exc)
            return

        broker_map: dict[str, int] = {}
        for bp in broker_positions:
            sym = bp.get("tradingsymbol", "")
            qty = int(bp.get("quantity", 0))
            if qty != 0:
                broker_map[sym] = qty

        for ticker, pos in self.positions.items():
            expected_qty = int((pos.weight * self.portfolio_value) / pos.entry_price) if pos.entry_price > 0 else 0
            broker_qty = broker_map.pop(ticker, 0)

            if expected_qty == 0:
                continue

            diff_pct = abs(broker_qty - expected_qty) / expected_qty if expected_qty else 1.0
            if diff_pct > 0.05:
                msg = (
                    f"Position mismatch: {ticker} internal={expected_qty} "
                    f"broker={broker_qty} (diff={diff_pct:.1%})"
                )
                logger.warning(msg)
                self.notifier.notify_error(f"RECONCILIATION: {msg}")
            elif broker_qty != expected_qty:
                logger.info("Minor qty diff: %s internal=%d broker=%d", ticker, expected_qty, broker_qty)

        for sym, qty in broker_map.items():
            logger.warning("Broker has position not in internal state: %s qty=%d", sym, qty)
            self.notifier.notify_error(f"RECONCILIATION: Unknown broker position {sym} qty={qty}")

    # ------------------------------------------------------------------
    # Token validity guard
    # ------------------------------------------------------------------

    def _require_valid_token(self, method_name: str) -> bool:
        """Return True if the token is valid or we're in dry-run mode."""
        if self.order_mgr.dry_run:
            return True
        if not self._token_valid:
            logger.error("Kite token invalid -- skipping %s", method_name)
            self.notifier.notify_error("HALTED: Kite token expired. All jobs suspended.")
            return False
        return True

    # ------------------------------------------------------------------
    # refresh_token  (09:10 IST)
    # ------------------------------------------------------------------

    def refresh_token(self) -> None:
        logger.info("Running token refresh check...")
        result = self.token_mgr.refresh_token()
        if result is not None:
            self._token_valid = bool(result)
            if not self._token_valid:
                self.notifier.notify_error("ALERT: Kite token refresh failed. Manual login required.")

    # ------------------------------------------------------------------
    # monitor_stops  (every 5 min, 09:15-15:30)
    # ------------------------------------------------------------------

    def monitor_stops(self) -> None:
        if not self._require_valid_token("monitor_stops"):
            return
        if not self.positions:
            return
        self._sync_portfolio_value()
        logger.info("Monitoring stops for %d positions...", len(self.positions))

        tickers = list(self.positions.keys())
        live_prices = get_live_quotes(tickers)
        if not live_prices:
            logger.warning("No live prices received")
            return

        today = datetime.now().date()
        if self._atr_cache_date != today or not self._atr_cache:
            self._atr_cache.clear()
            for ticker in tickers:
                df = fetch_historical(ticker, days=60, use_cache=True)
                if not df.empty and len(df) >= 14:
                    atr_val = compute_atr_series(df["high"], df["low"], df["close"], 14)
                    if not atr_val.empty and pd.notna(atr_val.iloc[-1]):
                        self._atr_cache[ticker] = float(atr_val.iloc[-1])
            self._atr_cache_date = today
        atr_values = self._atr_cache

        now = datetime.now()
        cache_time, cached_regime_level = self._regime_cache
        if cache_time is None or (now - cache_time).total_seconds() > 1800:
            bench_sym = self.cfg.get("strategy", {}).get("benchmark", "NIFTY 200").replace("NSE:", "")
            bench_df = fetch_historical(bench_sym, days=400, use_cache=True)
            if bench_df.empty:
                regime_level = RegimeLevel.RISK_OFF
            else:
                all_close_dict: dict[str, pd.Series] = {}
                for t in tickers:
                    df = fetch_historical(t, 400, use_cache=True)
                    if not df.empty and "close" in df.columns:
                        all_close_dict[t] = df["close"]
                all_close = pd.DataFrame(all_close_dict)
                if not all_close.empty:
                    regime = assess_regime(bench_df["close"], all_close, self.cfg)
                    regime_level = regime.level
                else:
                    regime_level = RegimeLevel.RISK_OFF
            self._regime_cache = (now, regime_level)
        else:
            regime_level = cached_regime_level

        snap_regime = type("R", (), {"level": regime_level})()
        cb_triggered = self.circuit_breaker.check(
            self.portfolio_value, self.portfolio_peak, snap_regime
        )

        if cb_triggered and not self.circuit_breaker.active:
            pass

        if self.circuit_breaker.active:
            logger.warning("Circuit breaker ACTIVE -- liquidating all positions")
            dd = (self.portfolio_value / self.portfolio_peak) - 1 if self.portfolio_peak > 0 else 0
            trades = self.order_mgr.liquidate_all(self.positions, live_prices, self.portfolio_value)
            for t in trades:
                self.state.record_trade(t)
                self.state.close_position(t.ticker)
            self.positions.clear()
            self.notifier.notify_circuit_breaker(dd, "TRIGGERED")
            self._save_snapshot()
            return

        exits, self.positions = check_all_stops(
            self.positions, live_prices, atr_values, self.cfg,
        )

        for sig in exits:
            trade_records = self.order_mgr.place_orders([sig], self.portfolio_value)
            for tr in trade_records:
                pos = self.positions.get(sig.ticker)
                if pos:
                    tr.holding_days = (datetime.now() - pos.entry_date).days
                self.state.record_trade(tr)

            self.state.close_position(sig.ticker)
            self.state.save_stopped_out(sig.ticker, sig.current_weight, datetime.now())
            self.stopped_out[sig.ticker] = sig.current_weight
            if sig.ticker in self.positions:
                del self.positions[sig.ticker]

            self.notifier.notify_stop_triggered(sig)

        self.state.update_position_prices(live_prices)
        self._last_stop_check = datetime.now()

    # ------------------------------------------------------------------
    # check_re_entry  (15:20 IST)
    # ------------------------------------------------------------------

    def check_re_entry_job(self) -> None:
        if not self._require_valid_token("check_re_entry_job"):
            return
        if not self.stopped_out:
            return
        self._refresh_rankings_if_stale()
        logger.info("Checking re-entry for %d stopped-out stocks...", len(self.stopped_out))

        tickers = list(self.stopped_out.keys())
        live_prices = get_live_quotes(tickers)

        dma_20: dict[str, float] = {}
        atr_values: dict[str, float] = {}
        for ticker in tickers:
            df = fetch_historical(ticker, days=60, use_cache=True)
            if df.empty or len(df) < 20:
                continue
            ma20 = sma(df["close"], 20)
            if not ma20.empty and pd.notna(ma20.iloc[-1]):
                dma_20[ticker] = float(ma20.iloc[-1])
            atr_val = compute_atr_series(df["high"], df["low"], df["close"], 14)
            if not atr_val.empty and pd.notna(atr_val.iloc[-1]):
                atr_values[ticker] = float(atr_val.iloc[-1])

        top_n = self.cfg.get("strategy", {}).get("top_momentum_n", 10)
        re_entry_signals = check_re_entry(
            self.stopped_out, self._last_ranked, live_prices,
            dma_20, atr_values, self.positions, top_n, self.cfg,
        )

        for sig in re_entry_signals:
            atr_mult = self.cfg.get("exits", {}).get("atr_multiple", 2.5)
            stop = compute_initial_stop(sig.price, atr_values.get(sig.ticker, 0), atr_mult)

            trade_records = self.order_mgr.place_orders([sig], self.portfolio_value)
            for tr in trade_records:
                self.state.record_trade(tr)

            new_pos = Position(
                ticker=sig.ticker,
                weight=sig.target_weight,
                entry_price=sig.price,
                entry_date=datetime.now(),
                high_watermark=sig.price,
                stop_price=stop,
                sector=SECTOR_MAP.get(sig.ticker, "Other"),
            )
            self.positions[sig.ticker] = new_pos
            self.state.save_position(new_pos)

            if sig.ticker in self.stopped_out:
                del self.stopped_out[sig.ticker]

            self.notifier.notify_re_entry(sig)

    # ------------------------------------------------------------------
    # daily_rebalance  (15:45 IST, last trading day)
    # ------------------------------------------------------------------

    def daily_rebalance(self) -> None:
        if not self._require_valid_token("daily_rebalance"):
            return
        if not _is_last_trading_day():
            logger.info("Not last trading day of month, skipping rebalance")
            self._save_snapshot()
            return
        if self.state.has_rebalanced_today():
            logger.warning("Rebalance already executed today, skipping duplicate")
            return

        self._sync_portfolio_value()
        self._reconcile_positions()
        logger.info("Running monthly rebalance...")

        all_tickers = BROAD_UNIVERSE.copy()
        bench_sym = self.cfg.get("strategy", {}).get("benchmark", "NIFTY 200").replace("NSE:", "")

        data = fetch_historical_bulk(
            all_tickers + [bench_sym], days=400, benchmark_col=bench_sym,
        )
        if not data or data["close"].empty:
            logger.error("No data available for rebalance")
            self.notifier.notify_error("Rebalance failed: no market data")
            return

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        if bench_sym not in close.columns:
            logger.error("Benchmark %s not found in data", bench_sym)
            self.notifier.notify_error(f"Rebalance failed: benchmark {bench_sym} missing")
            return

        benchmark_close = close[bench_sym].dropna()
        stocks_close = close.drop(columns=[bench_sym], errors="ignore")
        stocks_high = high.drop(columns=[bench_sym], errors="ignore")
        stocks_low = low.drop(columns=[bench_sym], errors="ignore")
        stocks_volume = volume.drop(columns=[bench_sym], errors="ignore")

        result = generate_rebalance_signals(
            stocks_close, stocks_high, stocks_low, stocks_volume,
            benchmark_close, self.positions, self.cfg,
        )

        result.target_weights = enforce_sector_caps(
            result.target_weights,
            self.cfg.get("sizing", {}).get("max_sector_weight", 0.30),
        )

        self.state.save_regime(result.regime)

        all_signals = result.sells + result.buys
        trade_records = self.order_mgr.place_orders(all_signals, self.portfolio_value)

        for sig in result.sells:
            if sig.ticker in self.positions:
                del self.positions[sig.ticker]
            self.state.close_position(sig.ticker)

        atr_mult = self.cfg.get("exits", {}).get("atr_multiple", 2.5)
        atr_df = compute_atr_df(stocks_high, stocks_low, stocks_close, 14)

        for sig in result.buys:
            atr_val = float(atr_df[sig.ticker].iloc[-1]) if sig.ticker in atr_df.columns and pd.notna(atr_df[sig.ticker].iloc[-1]) else 0.0
            stop = compute_initial_stop(sig.price, atr_val, atr_mult)
            tw = result.target_weights.get(sig.ticker, sig.target_weight)

            if sig.ticker in self.positions:
                self.positions[sig.ticker].weight = tw
            else:
                new_pos = Position(
                    ticker=sig.ticker,
                    weight=tw,
                    entry_price=sig.price,
                    entry_date=datetime.now(),
                    high_watermark=sig.price,
                    stop_price=stop,
                    sector=SECTOR_MAP.get(sig.ticker, "Other"),
                )
                self.positions[sig.ticker] = new_pos

            self.state.save_position(self.positions[sig.ticker])

        for tr in trade_records:
            self.state.record_trade(tr)

        daily_returns = stocks_close.pct_change().fillna(0)
        vol_60 = rolling_volatility(daily_returns, 60)
        adv = adv_126(stocks_close, stocks_volume)
        dma_100 = sma(stocks_close, 100)
        dma_200 = sma(stocks_close, 200)
        self._last_ranked = score_and_rank(
            stocks_close, stocks_volume, vol_60, adv,
            dma_100, dma_200, SECTOR_MAP, self.cfg,
        )

        exposure_signals = apply_exposure_floor(
            self.positions, self._last_ranked, stocks_close, atr_df, self.cfg,
        )
        if exposure_signals:
            floor_trades = self.order_mgr.place_orders(exposure_signals, self.portfolio_value)
            for sig in exposure_signals:
                atr_val = float(atr_df[sig.ticker].iloc[-1]) if sig.ticker in atr_df.columns and pd.notna(atr_df[sig.ticker].iloc[-1]) else 0.0
                stop = compute_initial_stop(sig.price, atr_val, atr_mult)
                new_pos = Position(
                    ticker=sig.ticker,
                    weight=sig.target_weight,
                    entry_price=sig.price,
                    entry_date=datetime.now(),
                    high_watermark=sig.price,
                    stop_price=stop,
                    sector=SECTOR_MAP.get(sig.ticker, "Other"),
                )
                self.positions[sig.ticker] = new_pos
                self.state.save_position(new_pos)
            for tr in floor_trades:
                self.state.record_trade(tr)

        self.stopped_out.clear()
        self.state.clear_stopped_out_month(datetime.now().strftime("%Y-%m"))

        self.state.save_rebalance(
            result.date, result.num_selected,
            result.regime.allocation_pct, result.target_weights,
        )

        self.notifier.notify_rebalance(result.buys, result.sells, result.regime)
        self._save_snapshot()
        self._last_rebalance = datetime.now()
        logger.info("Rebalance complete: %d buys, %d sells", len(result.buys), len(result.sells))

    # ------------------------------------------------------------------
    # check_pyramid  (15:00 IST daily, skip rebalance day)
    # ------------------------------------------------------------------

    def check_pyramid(self) -> None:
        """Add to winning positions that remain in the top-N momentum ranking."""
        if not self._require_valid_token("check_pyramid"):
            return
        pyramid_cfg = self.cfg.get("pyramid", {})
        if not pyramid_cfg.get("enabled", True):
            return
        if not self.positions or self._last_ranked.empty:
            return
        if _is_last_trading_day():
            return

        threshold = pyramid_cfg.get("threshold_pct", 0.05)
        add_pct = pyramid_cfg.get("add_pct", 0.03)
        max_pyramids = pyramid_cfg.get("max_pyramids", 2)
        ratchet = pyramid_cfg.get("ratchet_stop_to_breakeven", True)
        max_weight = self.cfg.get("sizing", {}).get("max_weight_per_stock", 0.18)
        top_n = self.cfg.get("strategy", {}).get("top_momentum_n", 10)

        top_ranked = set(self._last_ranked.head(top_n).index)
        live_prices = get_live_quotes(list(self.positions.keys()))
        if not live_prices:
            return

        for ticker, pos in list(self.positions.items()):
            price = live_prices.get(ticker)
            if price is None:
                continue
            if pos.pyramid_count >= max_pyramids:
                continue
            if pos.weight + add_pct > max_weight:
                continue
            gain = (price / pos.entry_price) - 1 if pos.entry_price > 0 else 0.0
            if gain >= threshold and ticker in top_ranked:
                pos.pyramid_count += 1
                pos.weight += add_pct

                if ratchet:
                    be_stop = pos.entry_price
                    if pos.stop_price is None or pos.stop_price < be_stop:
                        pos.stop_price = be_stop

                sig = Signal(
                    ticker=ticker,
                    action=OrderAction.BUY,
                    target_weight=pos.weight,
                    current_weight=pos.weight - add_pct,
                    price=price,
                    reason=f"Pyramid #{pos.pyramid_count} (gain {gain:.1%})",
                )
                trade_records = self.order_mgr.place_orders([sig], self.portfolio_value)
                for tr in trade_records:
                    self.state.record_trade(tr)
                self.state.save_position(pos)
                self.notifier.notify_re_entry(sig)
                logger.info(
                    "PYRAMID #%d: %s at %.2f (gain %.1f%%)",
                    pos.pyramid_count, ticker, price, gain * 100,
                )

    # ------------------------------------------------------------------
    # daily_summary  (16:00 IST)
    # ------------------------------------------------------------------

    def daily_summary(self) -> None:
        logger.info("Sending daily summary...")
        self._sync_portfolio_value()
        snap = self._current_snapshot()
        prev = self.state.load_latest_portfolio_state()
        day_pnl = snap.portfolio_value - (prev.portfolio_value if prev else self.initial_capital)
        self.notifier.notify_daily_summary(snap, day_pnl)
        self._save_snapshot()

    # ------------------------------------------------------------------
    # Periodic reports  (weekly / monthly / yearly)
    # ------------------------------------------------------------------

    def _send_report(self, period: str) -> None:
        from report_generator import generate_report

        try:
            report_path = generate_report(period, state_manager=self.state)
            caption = f"<b>{period.upper()} REPORT</b>"
            sent = self.notifier.send_document(report_path, caption)
            if sent:
                logger.info("%s report sent via Telegram", period.capitalize())
            else:
                logger.warning("%s report generated but Telegram send failed", period.capitalize())
        except Exception as exc:
            logger.error("Failed to generate %s report: %s", period, exc)
            self.notifier.notify_error(f"{period.capitalize()} report generation failed: {exc}")
        finally:
            try:
                report_path.unlink(missing_ok=True)
            except Exception:
                pass

    def weekly_report(self) -> None:
        """Generate and send the weekly trading report (Saturday 10:00 IST)."""
        logger.info("Generating weekly report...")
        self._send_report("weekly")

    def monthly_report(self) -> None:
        """Generate and send the monthly trading report (1st of month 10:00 IST)."""
        logger.info("Generating monthly report...")
        self._send_report("monthly")

    def yearly_report(self) -> None:
        """Generate and send the yearly trading report (Jan 1 10:00 IST)."""
        logger.info("Generating yearly report...")
        self._send_report("yearly")

    # ------------------------------------------------------------------
    # Status helpers (for CLI / API)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        pos_list = []
        for ticker, pos in self.positions.items():
            pos_list.append({
                "ticker": ticker,
                "weight": pos.weight,
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "high_watermark": pos.high_watermark,
                "sector": pos.sector,
            })

        exposure = sum(p.weight for p in self.positions.values())
        return {
            "portfolio_value": self.portfolio_value,
            "portfolio_peak": self.portfolio_peak,
            "circuit_breaker_active": self.circuit_breaker.active,
            "exposure": exposure,
            "positions_count": len(self.positions),
            "positions": pos_list,
        }

    def get_stops(self) -> list[dict]:
        return [
            {
                "ticker": t,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
                "high_watermark": p.high_watermark,
                "pct_from_stop": ((p.stop_price / p.high_watermark) - 1) * 100 if p.stop_price and p.high_watermark else None,
            }
            for t, p in self.positions.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_snapshot(self) -> PortfolioSnapshot:
        exposure = sum(p.weight for p in self.positions.values())
        return PortfolioSnapshot(
            date=datetime.now(),
            portfolio_value=self.portfolio_value,
            portfolio_peak=self.portfolio_peak,
            circuit_breaker_active=self.circuit_breaker.active,
            cash_weight=max(0.0, 1 - exposure),
            exposure=exposure,
            positions_count=len(self.positions),
        )

    def _save_snapshot(self) -> None:
        self.state.save_portfolio_state(self._current_snapshot())

    def _refresh_rankings_if_stale(self) -> None:
        """Re-run score_and_rank if rankings are empty or older than 7 days."""
        from datetime import timedelta

        if (
            not self._last_ranked.empty
            and self._last_ranked_date
            and (datetime.now() - self._last_ranked_date) < timedelta(days=7)
        ):
            return

        logger.info("Refreshing momentum rankings (stale or empty)...")
        try:
            all_tickers = BROAD_UNIVERSE.copy()
            bench_sym = self.cfg.get("strategy", {}).get("benchmark", "NIFTY 200").replace("NSE:", "")
            data = fetch_historical_bulk(all_tickers + [bench_sym], days=400, benchmark_col=bench_sym)
            if not data or data["close"].empty:
                logger.warning("Cannot refresh rankings: no data")
                return

            close = data["close"]
            volume = data["volume"]
            stocks_close = close.drop(columns=[bench_sym], errors="ignore")
            stocks_volume = volume.drop(columns=[bench_sym], errors="ignore")

            daily_returns = stocks_close.pct_change().fillna(0)
            vol_60 = rolling_volatility(daily_returns, 60)
            adv = adv_126(stocks_close, stocks_volume)
            dma_100 = sma(stocks_close, 100)
            dma_200 = sma(stocks_close, 200)

            self._last_ranked = score_and_rank(
                stocks_close, stocks_volume, vol_60, adv,
                dma_100, dma_200, SECTOR_MAP, self.cfg,
            )
            self._last_ranked_date = datetime.now()
            logger.info("Rankings refreshed: %d stocks scored", len(self._last_ranked))
        except Exception as exc:
            logger.error("Failed to refresh rankings: %s", exc)

    def force_rebalance(self) -> None:
        """Run rebalance logic without the last-trading-day calendar guard.

        Intended for CLI / manual override use only.
        """
        if not self._require_valid_token("force_rebalance"):
            return
        if self.state.has_rebalanced_today():
            logger.warning("Rebalance already executed today, skipping duplicate")
            return

        self._sync_portfolio_value()
        self._reconcile_positions()
        logger.info("Running FORCED monthly rebalance (calendar guard bypassed)...")

        all_tickers = BROAD_UNIVERSE.copy()
        bench_sym = self.cfg.get("strategy", {}).get("benchmark", "NIFTY 200").replace("NSE:", "")

        data = fetch_historical_bulk(
            all_tickers + [bench_sym], days=400, benchmark_col=bench_sym,
        )
        if not data or data["close"].empty:
            logger.error("No data available for rebalance")
            self.notifier.notify_error("Forced rebalance failed: no market data")
            return

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        if bench_sym not in close.columns:
            logger.error("Benchmark %s not found in data", bench_sym)
            self.notifier.notify_error(f"Forced rebalance failed: benchmark {bench_sym} missing")
            return

        benchmark_close = close[bench_sym].dropna()
        stocks_close = close.drop(columns=[bench_sym], errors="ignore")
        stocks_high = high.drop(columns=[bench_sym], errors="ignore")
        stocks_low = low.drop(columns=[bench_sym], errors="ignore")
        stocks_volume = volume.drop(columns=[bench_sym], errors="ignore")

        result = generate_rebalance_signals(
            stocks_close, stocks_high, stocks_low, stocks_volume,
            benchmark_close, self.positions, self.cfg,
        )

        result.target_weights = enforce_sector_caps(
            result.target_weights,
            self.cfg.get("sizing", {}).get("max_sector_weight", 0.30),
        )

        self.state.save_regime(result.regime)

        all_signals = result.sells + result.buys
        trade_records = self.order_mgr.place_orders(all_signals, self.portfolio_value)

        for sig in result.sells:
            if sig.ticker in self.positions:
                del self.positions[sig.ticker]
            self.state.close_position(sig.ticker)

        atr_mult = self.cfg.get("exits", {}).get("atr_multiple", 2.5)
        atr_df = compute_atr_df(stocks_high, stocks_low, stocks_close, 14)

        for sig in result.buys:
            atr_val = float(atr_df[sig.ticker].iloc[-1]) if sig.ticker in atr_df.columns and pd.notna(atr_df[sig.ticker].iloc[-1]) else 0.0
            stop = compute_initial_stop(sig.price, atr_val, atr_mult)
            tw = result.target_weights.get(sig.ticker, sig.target_weight)

            if sig.ticker in self.positions:
                self.positions[sig.ticker].weight = tw
            else:
                new_pos = Position(
                    ticker=sig.ticker,
                    weight=tw,
                    entry_price=sig.price,
                    entry_date=datetime.now(),
                    high_watermark=sig.price,
                    stop_price=stop,
                    sector=SECTOR_MAP.get(sig.ticker, "Other"),
                )
                self.positions[sig.ticker] = new_pos

            self.state.save_position(self.positions[sig.ticker])

        for tr in trade_records:
            self.state.record_trade(tr)

        self.stopped_out.clear()
        self.state.clear_stopped_out_month(datetime.now().strftime("%Y-%m"))

        self.state.save_rebalance(
            result.date, result.num_selected,
            result.regime.allocation_pct, result.target_weights,
        )

        self.notifier.notify_rebalance(result.buys, result.sells, result.regime)
        self._save_snapshot()
        self._last_rebalance = datetime.now()
        logger.info("Forced rebalance complete: %d buys, %d sells", len(result.buys), len(result.sells))

    def get_health(self) -> dict:
        """Return health metrics for the /health endpoint."""
        db_ok = False
        try:
            self.state.load_latest_portfolio_state()
            db_ok = True
        except Exception:
            pass

        exposure = sum(p.weight for p in self.positions.values())
        return {
            "db_connected": db_ok,
            "token_valid": self._token_valid,
            "last_stop_check": self._last_stop_check.isoformat() if self._last_stop_check else None,
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "portfolio_value": self.portfolio_value,
            "exposure": exposure,
            "positions_count": len(self.positions),
            "circuit_breaker_active": self.circuit_breaker.active,
        }
