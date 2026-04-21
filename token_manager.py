"""Daily Kite access-token refresh flow.

Kite access tokens expire every day. This module validates the current
token, handles the request-token -> access-token exchange, and alerts
via Telegram when the token is expired.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml

from notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def _load_broker_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    broker = cfg.get("broker", {})
    resolved: dict = {}
    for k, v in broker.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            resolved[k] = os.environ.get(v[2:-1], "")
        else:
            resolved[k] = v
    return resolved


class TokenManager:
    """Manage Kite Connect access token lifecycle."""

    def __init__(self):
        self.broker_cfg = _load_broker_config()
        self.api_key = self.broker_cfg.get("api_key", "")
        self.api_secret = self.broker_cfg.get("api_secret", "")
        self.token_file = Path(self.broker_cfg.get("access_token_file", "kite_token.json"))
        self.notifier = TelegramNotifier()

    def _get_kite(self):
        from kiteconnect import KiteConnect  # type: ignore[import-untyped]
        kc = KiteConnect(api_key=self.api_key)
        token = self._load_token()
        if token:
            kc.set_access_token(token)
        return kc

    def _load_token(self) -> str:
        if self.token_file.exists():
            with open(self.token_file) as f:
                data = json.load(f)
            return data.get("access_token", "")
        return ""

    def _save_token(self, access_token: str) -> None:
        with open(self.token_file, "w") as f:
            json.dump({"access_token": access_token}, f)
        logger.info("Access token saved to %s", self.token_file)

    def validate_token(self) -> bool:
        """Check if the current access token is valid by calling kite.profile()."""
        token = self._load_token()
        if not token:
            logger.warning("No access token found")
            return False
        try:
            kite = self._get_kite()
            profile = kite.profile()
            logger.info("Token valid for user: %s", profile.get("user_name", "unknown"))
            return True
        except Exception as exc:
            logger.warning("Token validation failed: %s", exc)
            return False

    def refresh_token(self) -> bool:
        """Validate the token; if invalid, send Telegram alert.

        Returns True if the token is valid, False if expired/invalid.
        """
        if self.validate_token():
            return True

        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"
        logger.warning("Token expired. Login URL: %s", login_url)
        self.notifier.notify_token_expiry(login_url)
        return False

    def exchange_request_token(self, request_token: str) -> bool:
        """Exchange a request_token for an access_token and persist it.

        Called manually via CLI after user logs in via browser.
        """
        try:
            from kiteconnect import KiteConnect  # type: ignore[import-untyped]
            kc = KiteConnect(api_key=self.api_key)
            data = kc.generate_session(request_token, api_secret=self.api_secret)
            access_token = data.get("access_token", "")
            if not access_token:
                logger.error("generate_session returned no access_token")
                return False
            self._save_token(access_token)

            from data_manager import reset_kite
            reset_kite()

            logger.info("Token exchange successful")
            return True
        except Exception as exc:
            logger.error("Token exchange failed: %s", exc)
            return False

    def get_login_url(self) -> str:
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"
