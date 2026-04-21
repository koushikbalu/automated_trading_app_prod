"""Zerodha Kite Connect order placement, status tracking, retry logic,
and DRY_RUN mode.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
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

        order_id: Optional[str] = None
        fill_price = signal.price
        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s qty=%d @ %.2f (weight=%.3f, reason=%s)",
                txn_type, signal.ticker, qty, signal.price,
                signal.target_weight, signal.reason,
            )
        else:
            order_id = self._place_with_retry(signal.ticker, txn_type, qty, signal.price)
            if order_id:
                try:
                    actual_price, filled_qty, status = self.wait_for_fill(order_id)
                    if status == "COMPLETE" and actual_price is not None:
                        fill_price = actual_price
                    elif status == "REJECTED":
                        logger.error("Order REJECTED: %s %s order_id=%s", txn_type, signal.ticker, order_id)
                        if self._notifier:
                            self._notifier.notify_error(
                                f"Order REJECTED: {txn_type} {signal.ticker} (order_id={order_id})"
                            )
                    elif status == "PARTIAL":
                        logger.warning(
                            "PARTIAL fill: %s %s filled_qty=%d order_id=%s",
                            txn_type, signal.ticker, filled_qty, order_id,
                        )
                except OrderTimeoutError:
                    logger.warning("Fill confirmation timed out for %s order_id=%s", signal.ticker, order_id)

        cost = abs(signal.target_weight - signal.current_weight) * (self.one_way_cost + self.slippage)

        return TradeRecord(
            date=datetime.now(),
            ticker=signal.ticker,
            action=txn_type,
            price=fill_price,
            weight_traded=abs(signal.target_weight - signal.current_weight),
            reason=signal.reason,
            order_id=order_id,
            costs=cost,
        )

    def place_orders(
        self,
        signals: list[Signal],
        portfolio_value: float,
    ) -> list[TradeRecord]:
        """Place orders for a list of signals. Sells first, then buys."""
        sell_signals = [s for s in signals if s.action in (OrderAction.SELL, OrderAction.REBAL_EXIT)]
        buy_signals = [s for s in signals if s.action == OrderAction.BUY]

        records: list[TradeRecord] = []
        for sig in sell_signals:
            records.append(self.place_order(sig, portfolio_value))
        for sig in buy_signals:
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
                limit_price = round(price * (1 - self.limit_buffer_pct), 2)
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
                logger.warning(
                    "Order attempt %d failed for %s %s: %s",
                    attempt, txn_type, symbol, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        logger.error("All %d order attempts failed for %s %s", self.max_retries, txn_type, symbol)
        return None
