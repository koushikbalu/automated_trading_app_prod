"""Zerodha Kite Connect order placement, status tracking, retry logic,
and DRY_RUN mode.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any, Optional

import yaml
from pathlib import Path

from data_manager import get_kite, get_exchange, _rate_limit
from models import OrderAction, Signal, TradeRecord

logger = logging.getLogger(__name__)


class OrderTimeoutError(Exception):
    """Raised when wait_for_fill exceeds the configured timeout."""


class OrderRejectedError(Exception):
    """Raised when the exchange rejects an order."""


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class OrderManager:
    """Place, track, and retry orders via Kite Connect.

    In DRY_RUN mode (default), orders are logged but not sent to the
    exchange.
    """

    def __init__(self, cfg: dict | None = None, notifier: Any = None):
        if cfg is None:
            cfg = _load_config()
        broker = cfg.get("broker", {})
        self.dry_run: bool = broker.get("dry_run", True)
        self.exchange: str = broker.get("exchange", "NSE")
        self.product_type: str = broker.get("product_type", "CNC")
        self.order_type: str = broker.get("order_type", "MARKET")
        self.limit_buffer_pct: float = broker.get("limit_buffer_pct", 0.002)
        self.max_retries: int = 3
        self.retry_delay: float = 2.0
        self._notifier = notifier

        costs = cfg.get("costs", {})
        self.one_way_cost = costs.get("one_way_brokerage", 0.0003)
        self.slippage = costs.get("slippage_estimate", 0.001)

        self._orders_placed_today: set[tuple[str, str, str]] = set()
        self._orders_placed_date: date | None = None

        self.cfg = cfg
        exec_cfg = cfg.get("execution", {})
        self.max_participation_rate: float = exec_cfg.get("max_participation_rate", 0.05)
        self.max_slice_value: float = exec_cfg.get("max_slice_value", 2_500_000)
        self.slice_delay: float = exec_cfg.get("slice_delay_seconds", 30)
        self.freeze_qty_buffer: float = exec_cfg.get("freeze_qty_buffer", 0.90)

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        signal: Signal,
        portfolio_value: float,
    ) -> TradeRecord:
        """Place a single order based on a Signal. Returns a TradeRecord."""
        qty = self._weight_to_qty(signal.target_weight, signal.price, portfolio_value)
        if qty <= 0:
            qty = self._weight_to_qty(signal.current_weight, signal.price, portfolio_value)
            if qty <= 0:
                qty = 1

        if signal.action in (OrderAction.SELL, OrderAction.REBAL_EXIT):
            txn_type = "SELL"
        else:
            txn_type = "BUY"

        today = datetime.now().date()
        if self._orders_placed_date != today:
            self._orders_placed_today.clear()
            self._orders_placed_date = today
        dedup_key = (signal.ticker, str(today), txn_type)
        if dedup_key in self._orders_placed_today:
            logger.warning("Duplicate order blocked: %s %s", txn_type, signal.ticker)
            return TradeRecord(
                date=datetime.now(), ticker=signal.ticker, action=txn_type,
                price=signal.price, weight_traded=0.0, reason="DUPLICATE_BLOCKED",
                fill_status="SKIPPED",
            )

        order_id: Optional[str] = None
        fill_price = signal.price
        realized_qty = qty
        fill_status = "UNKNOWN"

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%d @ %.2f (weight=%.3f, reason=%s)",
                txn_type, signal.ticker, qty, signal.price,
                signal.target_weight, signal.reason,
            )
            realized_qty = qty
            fill_status = "DRY_RUN"
        else:
            order_id = self._place_with_retry(signal.ticker, txn_type, qty, signal.price, signal.reason)
            if order_id:
                try:
                    actual_price, filled_qty, status = self.wait_for_fill(order_id)
                    fill_status = status
                    if status == "COMPLETE" and actual_price is not None:
                        fill_price = actual_price
                        realized_qty = filled_qty
                    elif status == "PARTIAL":
                        fill_price = actual_price if actual_price else signal.price
                        realized_qty = filled_qty
                        self._cancel_pending(order_id)
                        if self._notifier:
                            self._notifier.notify_error(
                                f"PARTIAL FILL: {txn_type} {signal.ticker} "
                                f"filled {filled_qty}/{qty} (order_id={order_id})"
                            )
                    elif status == "REJECTED":
                        realized_qty = 0
                        logger.error("Order REJECTED: %s %s order_id=%s", txn_type, signal.ticker, order_id)
                        if self._notifier:
                            self._notifier.notify_error(
                                f"Order REJECTED: {txn_type} {signal.ticker} (order_id={order_id})"
                            )
                except OrderTimeoutError:
                    fill_status = "TIMEOUT"
                    logger.warning("Fill confirmation timed out for %s order_id=%s", signal.ticker, order_id)
            else:
                fill_status = "REJECTED"
                realized_qty = 0

        realized_weight = (realized_qty * fill_price) / portfolio_value if portfolio_value > 0 else 0.0
        cost = realized_weight * (self.one_way_cost + self.slippage)

        self._orders_placed_today.add(dedup_key)

        return TradeRecord(
            date=datetime.now(),
            ticker=signal.ticker,
            action=txn_type,
            price=fill_price,
            weight_traded=realized_weight,
            reason=signal.reason,
            order_id=order_id,
            costs=cost,
            filled_qty=realized_qty,
            requested_qty=qty,
            fill_status=fill_status,
        )

    def place_orders(
        self,
        signals: list[Signal],
        portfolio_value: float,
    ) -> list[TradeRecord]:
        """Place orders for a list of signals. Sells first, then buys.

        Orders whose notional value exceeds max_slice_value are
        automatically routed through place_order_sliced().
        """
        sell_signals = [s for s in signals if s.action in (OrderAction.SELL, OrderAction.REBAL_EXIT)]
        buy_signals = [s for s in signals if s.action == OrderAction.BUY]

        records: list[TradeRecord] = []
        for sig in sell_signals + buy_signals:
            order_value = sig.target_weight * portfolio_value
            if order_value > self.max_slice_value:
                records.extend(self.place_order_sliced(sig, portfolio_value))
            else:
                records.append(self.place_order(sig, portfolio_value))

        return records

    def liquidate_all(
        self,
        positions: dict[str, Any],
        live_prices: dict[str, float],
        portfolio_value: float,
    ) -> list[TradeRecord]:
        """Emergency: market sell all positions."""
        signals = []
        for ticker, pos in positions.items():
            price = live_prices.get(ticker, pos.entry_price)
            signals.append(Signal(
                ticker=ticker,
                action=OrderAction.SELL,
                target_weight=0.0,
                current_weight=pos.weight,
                price=price,
                reason="Circuit breaker liquidation",
            ))
        return self.place_orders(signals, portfolio_value)

    # ------------------------------------------------------------------
    # Fill confirmation
    # ------------------------------------------------------------------

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: int = 30,
        poll_interval: float = 2.0,
    ) -> tuple[Optional[float], int, str]:
        """Poll order history until filled, rejected, or timeout.

        Returns (fill_price, filled_qty, status) where status is one of
        COMPLETE, REJECTED, PARTIAL, or TIMEOUT.
        """
        if self.dry_run:
            return (None, 0, "DRY_RUN")

        kite = get_kite()
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            try:
                history = kite.order_history(order_id)
                if not history:
                    time.sleep(poll_interval)
                    continue

                last = history[-1]
                status = last.get("status", "").upper()

                if status == "COMPLETE":
                    return (
                        float(last.get("average_price", 0)),
                        int(last.get("filled_quantity", 0)),
                        "COMPLETE",
                    )
                if status in ("REJECTED", "CANCELLED"):
                    return (None, 0, "REJECTED")
                if status == "TRADED" and int(last.get("pending_quantity", 1)) > 0:
                    return (
                        float(last.get("average_price", 0)),
                        int(last.get("filled_quantity", 0)),
                        "PARTIAL",
                    )
            except Exception as exc:
                logger.warning("Error polling order %s: %s", order_id, exc)

            time.sleep(poll_interval)

        raise OrderTimeoutError(f"Order {order_id} did not fill within {timeout_seconds}s")

    # ------------------------------------------------------------------
    # Order status
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> dict:
        """Fetch order status from Kite."""
        if self.dry_run:
            return {"status": "DRY_RUN", "order_id": order_id}
        try:
            kite = get_kite()
            history = kite.order_history(order_id)
            if history:
                return history[-1]
            return {"status": "UNKNOWN"}
        except Exception as exc:
            logger.error("Order status check failed for %s: %s", order_id, exc)
            return {"status": "ERROR", "message": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def place_order_sliced(
        self,
        signal: Signal,
        portfolio_value: float,
        adv_value: float | None = None,
        freeze_qty: int | None = None,
    ) -> list[TradeRecord]:
        """Split large orders into slices based on ADV and freeze qty limits."""
        total_qty = self._weight_to_qty(signal.target_weight, signal.price, portfolio_value)
        if total_qty <= 0:
            total_qty = 1

        max_slice = self._compute_max_slice(signal.price, adv_value, freeze_qty)

        if total_qty <= max_slice:
            return [self.place_order(signal, portfolio_value)]

        slices = []
        remaining = total_qty
        while remaining > 0:
            chunk = min(remaining, max_slice)
            slices.append(chunk)
            remaining -= chunk

        logger.info(
            "Splitting %s %s qty=%d into %d slices (max_slice=%d)",
            signal.action.value if hasattr(signal.action, "value") else signal.action,
            signal.ticker, total_qty, len(slices), max_slice,
        )

        records: list[TradeRecord] = []
        filled_so_far = 0
        for i, slice_qty in enumerate(slices):
            slice_weight = (slice_qty * signal.price) / portfolio_value if portfolio_value > 0 else 0.0
            sub_signal = Signal(
                ticker=signal.ticker,
                action=signal.action,
                target_weight=signal.current_weight + slice_weight if signal.action == OrderAction.BUY else signal.current_weight - slice_weight,
                current_weight=signal.current_weight,
                price=signal.price,
                reason=f"{signal.reason} [slice {i+1}/{len(slices)}]",
            )
            tr = self.place_order(sub_signal, portfolio_value)
            records.append(tr)
            filled_so_far += tr.filled_qty

            if tr.fill_status in ("REJECTED", "TIMEOUT"):
                logger.warning("Slice %d/%d failed for %s -- aborting remaining slices", i+1, len(slices), signal.ticker)
                break

            if i < len(slices) - 1:
                time.sleep(self.slice_delay)

        return records

    def _compute_max_slice(self, price: float, adv_value: float | None, freeze_qty: int | None) -> int:
        """Compute max order qty from ADV participation, freeze limits, and value cap."""
        limits: list[int] = []

        if adv_value and adv_value > 0 and price > 0:
            limits.append(int(adv_value * self.max_participation_rate / price))

        if price > 0:
            limits.append(int(self.max_slice_value / price))

        if freeze_qty and freeze_qty > 0:
            limits.append(int(freeze_qty * self.freeze_qty_buffer))

        if not limits:
            return 50000
        return max(1, min(limits))

    def _cancel_pending(self, order_id: str) -> None:
        """Cancel remaining quantity on a partially filled order."""
        if self.dry_run:
            return
        try:
            kite = get_kite()
            _rate_limit()
            kite.cancel_order(variety="regular", order_id=order_id)
            logger.info("Cancelled pending remainder of order %s", order_id)
        except Exception as exc:
            logger.warning("Failed to cancel order %s: %s", order_id, exc)

    def _weight_to_qty(self, weight: float, price: float, portfolio_value: float) -> int:
        if price <= 0 or weight <= 0:
            return 0
        return int((weight * portfolio_value) / price)

    def _place_with_retry(
        self,
        symbol: str,
        txn_type: str,
        qty: int,
        price: float,
        reason: str = "",
    ) -> Optional[str]:
        """Attempt order placement with retries on transient failure."""
        kite = get_kite()
        exchange = get_exchange(symbol)

        order_params: dict[str, Any] = {
            "variety": "regular",
            "exchange": exchange,
            "tradingsymbol": symbol,
            "transaction_type": kite.TRANSACTION_TYPE_BUY if txn_type == "BUY" else kite.TRANSACTION_TYPE_SELL,
            "quantity": qty,
            "product": self.product_type,
            "order_type": kite.ORDER_TYPE_MARKET,
        }

        if self.order_type == "LIMIT":
            if txn_type == "BUY":
                limit_price = round(price * (1 + self.limit_buffer_pct), 2)
            else:
                extra_slippage = 0.0
                if reason and "stop" in reason.lower():
                    extra_slippage = self.cfg.get("exits", {}).get("stop_exit_slippage", 0.005)
                limit_price = round(price * (1 - self.limit_buffer_pct - extra_slippage), 2)
            order_params["order_type"] = kite.ORDER_TYPE_LIMIT
            order_params["price"] = limit_price

        for attempt in range(1, self.max_retries + 1):
            try:
                _rate_limit()
                order_id = kite.place_order(**order_params)
                logger.info(
                    "Order placed: %s %s qty=%d order_id=%s (attempt %d)",
                    txn_type, symbol, qty, order_id, attempt,
                )
                return order_id
            except Exception as exc:
                exc_type = type(exc).__name__
                non_retryable = ("InputException", "TokenException", "PermissionException")
                if exc_type in non_retryable:
                    logger.error("Non-retryable order error for %s %s: %s", txn_type, symbol, exc)
                    return None

                logger.warning(
                    "Transient order error (attempt %d) for %s %s: %s",
                    attempt, txn_type, symbol, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        logger.error("All %d order attempts failed for %s %s", self.max_retries, txn_type, symbol)
        return None
