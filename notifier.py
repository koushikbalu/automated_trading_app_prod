"""Telegram notification service with structured message templates.

Uses a background daemon thread with a queue for non-blocking sends.
"""

from __future__ import annotations

import logging
import os
import threading
from queue import Queue
from typing import Any

import httpx
import yaml
from pathlib import Path

from models import PortfolioSnapshot, RegimeState, Signal, TradeRecord

logger = logging.getLogger(__name__)


def _load_telegram_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tg = cfg.get("notifications", {}).get("telegram", {})
    resolved: dict[str, Any] = {}
    for k, v in tg.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            resolved[k] = os.environ.get(v[2:-1], "")
        else:
            resolved[k] = v
    return resolved


class TelegramNotifier:
    """Send Telegram messages for trading events via a non-blocking queue."""

    def __init__(self):
        tg = _load_telegram_config()
        self.bot_token: str = tg.get("bot_token", "")
        self.chat_id: str = tg.get("chat_id", "")
        self.send_on: list[str] = tg.get("send_on", [])
        self._queue: Queue = Queue()
        self._worker = threading.Thread(target=self._send_loop, daemon=True)
        self._worker.start()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _send_loop(self) -> None:
        """Background worker that drains the queue and sends messages."""
        while True:
            text = self._queue.get()
            try:
                self._send_sync(text)
            except Exception as exc:
                logger.error("Telegram background send error: %s", exc)
            finally:
                self._queue.task_done()

    def _send_sync(self, text: str) -> bool:
        """Synchronous send (called from background thread only)."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        for attempt in range(2):
            try:
                r = httpx.post(
                    url,
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10.0,
                )
                if r.status_code == 200:
                    return True
                logger.warning("Telegram send failed (attempt %d): %s", attempt + 1, r.text)
            except Exception as exc:
                logger.error("Telegram send error (attempt %d): %s", attempt + 1, exc)
        return False

    def _send(self, text: str) -> bool:
        """Non-blocking enqueue for all notification methods."""
        if not self.enabled:
            logger.debug("Telegram not configured, skipping notification")
            return False
        self._queue.put(text)
        return True

    # -- Event methods -----------------------------------------------------

    def notify_rebalance(self, buys: list[Signal], sells: list[Signal], regime: RegimeState) -> None:
        if "signal_generated" not in self.send_on:
            return
        lines = [f"<b>REBALANCE SIGNAL</b> ({regime.level.value}, alloc {regime.allocation_pct:.0%})"]
        if buys:
            lines.append("\n<b>BUY:</b>")
            for s in buys:
                lines.append(f"  {s.ticker} @ {s.target_weight:.1%} (₹{s.price:,.0f})")
        if sells:
            lines.append("\n<b>SELL:</b>")
            for s in sells:
                lines.append(f"  {s.ticker} ({s.reason})")
        if not buys and not sells:
            lines.append("No changes needed.")
        self._send("\n".join(lines))

    def notify_order(self, trade: TradeRecord) -> None:
        if "order_placed" not in self.send_on:
            return
        self._send(
            f"<b>ORDER {trade.action}</b>: {trade.ticker}\n"
            f"Price: ₹{trade.price:,.2f}\n"
            f"Weight: {trade.weight_traded:.1%}\n"
            f"Reason: {trade.reason}"
        )

    def notify_stop_triggered(self, signal: Signal, holding_days: int | None = None) -> None:
        if "stop_triggered" not in self.send_on:
            return
        msg = (
            f"<b>ATR STOP</b>: {signal.ticker}\n"
            f"Triggered at ₹{signal.price:,.2f}\n"
            f"Weight: {signal.current_weight:.1%}"
        )
        if holding_days is not None:
            msg += f"\nHeld: {holding_days} days"
        self._send(msg)

    def notify_re_entry(self, signal: Signal) -> None:
        if "re_entry" not in self.send_on:
            return
        self._send(
            f"<b>RE-ENTRY</b>: {signal.ticker}\n"
            f"Price: ₹{signal.price:,.2f} (above 20DMA)\n"
            f"Weight: {signal.target_weight:.1%}"
        )

    def notify_circuit_breaker(self, portfolio_dd: float, action: str) -> None:
        if "circuit_breaker" not in self.send_on:
            return
        self._send(
            f"<b>CIRCUIT BREAKER {action.upper()}</b>\n"
            f"Portfolio drawdown: {portfolio_dd:.1%}\n"
            f"All positions liquidated."
        )

    def notify_daily_summary(self, snap: PortfolioSnapshot, day_pnl: float) -> None:
        if "daily_summary" not in self.send_on:
            return
        self._send(
            f"<b>DAILY SUMMARY</b>\n"
            f"Portfolio: ₹{snap.portfolio_value:,.0f}\n"
            f"Day P&L: ₹{day_pnl:,.0f}\n"
            f"Exposure: {snap.exposure:.0%}\n"
            f"Positions: {snap.positions_count}\n"
            f"Peak: ₹{snap.portfolio_peak:,.0f}\n"
            f"Circuit Breaker: {'ACTIVE' if snap.circuit_breaker_active else 'OFF'}"
        )

    def notify_token_expiry(self, login_url: str = "") -> None:
        if "token_expiry" not in self.send_on:
            return
        msg = "<b>KITE TOKEN EXPIRED</b>\nPlease login to refresh."
        if login_url:
            msg += f"\n{login_url}"
        self._send(msg)

    def notify_error(self, message: str) -> None:
        if "error" not in self.send_on:
            return
        self._send(f"<b>ERROR</b>: {message}")

    def send_document(self, file_path: Path, caption: str = "") -> bool:
        """Upload a file via Telegram sendDocument (up to 50 MB)."""
        if not self.enabled:
            logger.debug("Telegram not configured, skipping document send")
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
        try:
            with open(file_path, "rb") as f:
                r = httpx.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"document": (file_path.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                    timeout=30.0,
                )
            if r.status_code != 200:
                logger.warning("Telegram sendDocument failed: %s", r.text)
                return False
            return True
        except Exception as exc:
            logger.error("Telegram sendDocument error: %s", exc)
            return False
