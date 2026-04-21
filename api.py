"""Lightweight FastAPI: health, status, report download, and Kite webhook endpoints.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse

from engine import TradingEngine
from token_manager import TokenManager
from report_generator import generate_report
from state_manager import StateManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Automated Trading System", version="1.0.0")

_engine: TradingEngine | None = None


def _get_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine


@app.get("/health")
def health():
    """Comprehensive health check: DB, token, last runs, portfolio."""
    engine = _get_engine()
    try:
        details = engine.get_health()
        overall = "ok" if details["db_connected"] and details["token_valid"] else "degraded"
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return {"status": "error", "timestamp": datetime.now().isoformat(), "error": str(exc)}

    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        **details,
    }


@app.get("/status")
def status():
    """Current portfolio value, positions, exposure, regime."""
    engine = _get_engine()
    return engine.get_status()


@app.get("/callback", response_class=HTMLResponse)
def kite_callback(request_token: str = Query(...), status: str = Query(default="")):
    """Kite OAuth redirect handler.

    Set your Kite Connect app's redirect URL to
    http://<host>:8000/callback in the Zerodha developer console.
    After browser login, Zerodha redirects here with ?request_token=...
    and the token is exchanged automatically.
    """
    if status == "error":
        logger.error("Kite login returned error status")
        return _callback_page(False, "Zerodha login was cancelled or failed.")

    tm = TokenManager()
    ok = tm.exchange_request_token(request_token)
    if ok:
        logger.info("Kite token exchanged via /callback")
        return _callback_page(True, "Token refreshed successfully. You can close this page.")
    logger.error("Kite token exchange failed via /callback")
    return _callback_page(False, "Token exchange failed. Check server logs.")


def _callback_page(success: bool, message: str) -> str:
    color = "#22c55e" if success else "#ef4444"
    icon = "&#10004;" if success else "&#10006;"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Kite Token</title></head>"
        f"<body style='font-family:system-ui;display:flex;justify-content:center;"
        f"align-items:center;height:100vh;margin:0;background:#0f172a;color:#e2e8f0'>"
        f"<div style='text-align:center'>"
        f"<div style='font-size:4rem;color:{color}'>{icon}</div>"
        f"<h2>{message}</h2></div></body></html>"
    )


@app.post("/webhook/kite")
async def kite_webhook(request: Request):
    """Kite postback for order updates (optional).

    Kite sends a POST with order status updates if configured
    in the Kite Connect dashboard.
    """
    try:
        body = await request.json()
        logger.info("Kite webhook received: %s", body)
        return {"status": "received"}
    except Exception as exc:
        logger.error("Kite webhook parse error: %s", exc)
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Report download endpoints
# ---------------------------------------------------------------------------

def _report_response(period: str, ref_date: Optional[str]) -> FileResponse:
    reference = date.fromisoformat(ref_date) if ref_date else None
    sm = StateManager()
    path = generate_report(period, reference_date=reference, state_manager=sm)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


@app.get("/reports/weekly")
def report_weekly(date: Optional[str] = Query(None, description="Reference date YYYY-MM-DD")):
    """Download the weekly trading report as an Excel file."""
    return _report_response("weekly", date)


@app.get("/reports/monthly")
def report_monthly(date: Optional[str] = Query(None, description="Reference date YYYY-MM-DD")):
    """Download the monthly trading report as an Excel file."""
    return _report_response("monthly", date)


@app.get("/reports/yearly")
def report_yearly(date: Optional[str] = Query(None, description="Reference date YYYY-MM-DD")):
    """Download the yearly trading report as an Excel file."""
    return _report_response("yearly", date)
