"""Entry point: Typer CLI + APScheduler.

CLI modes:
    python main.py run          # start scheduler (all jobs)
    python main.py rebalance    # run one-off rebalance now (bypasses calendar)
    python main.py status       # print current positions + P&L
    python main.py stops        # show all stop levels
    python main.py regime       # show current regime assessment
    python main.py history      # recent trades
    python main.py token        # manual Kite token refresh
    python main.py backtest     # run historical backtest using same config
    python main.py report       # generate a weekly/monthly/yearly Excel report
"""

from __future__ import annotations

import logging
import signal as signal_mod
import sys
from pathlib import Path

import typer
import yaml

app = typer.Typer(help="Automated Momentum Trading System")


def _setup_logging() -> None:
    cfg_path = Path(__file__).parent / "config.yaml"
    level = "INFO"
    log_file = "automated_trading.log"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        log_cfg = cfg.get("logging", {})
        level = log_cfg.get("level", "INFO")
        log_file = log_cfg.get("file", "automated_trading.log")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------

@app.command()
def run():
    """Start the scheduler with all scheduled jobs."""
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Automated Trading System scheduler...")

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.executors.pool import ThreadPoolExecutor

    from engine import TradingEngine

    cfg = _load_config()
    schedule = cfg.get("schedule", {})
    engine = TradingEngine(cfg)

    executors = {"default": ThreadPoolExecutor(max_workers=3)}
    scheduler = BlockingScheduler(
        timezone="Asia/Kolkata",
        executors=executors,
        job_defaults={"coalesce": True, "max_instances": 1},
    )

    token_time = schedule.get("token_refresh_time", "09:10")
    th, tm = token_time.split(":")
    scheduler.add_job(
        engine.refresh_token,
        CronTrigger(hour=int(th), minute=int(tm), day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="token_refresh",
        name="Token refresh",
    )

    interval_min = schedule.get("stop_monitor_interval_minutes", 5)
    market_open = schedule.get("market_open", "09:15")
    market_close = schedule.get("market_close", "15:30")
    oh, om = market_open.split(":")
    ch, cm = market_close.split(":")
    scheduler.add_job(
        engine.monitor_stops,
        CronTrigger(
            hour=f"{int(oh)}-{int(ch)}",
            minute=f"*/{interval_min}",
            day_of_week="mon-fri",
            timezone="Asia/Kolkata",
        ),
        id="stop_monitor",
        name="Stop monitor",
    )

    re_entry_time = schedule.get("re_entry_check_time", "15:20")
    rh, rm = re_entry_time.split(":")
    scheduler.add_job(
        engine.check_re_entry_job,
        CronTrigger(hour=int(rh), minute=int(rm), day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="re_entry_check",
        name="Re-entry check",
    )

    rebal_time = schedule.get("rebalance_time", "15:45")
    rbh, rbm = rebal_time.split(":")
    scheduler.add_job(
        engine.daily_rebalance,
        CronTrigger(hour=int(rbh), minute=int(rbm), day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="daily_rebalance",
        name="Daily rebalance",
    )

    pyramid_time = schedule.get("pyramid_check_time", "15:00")
    ph, pm = pyramid_time.split(":")
    scheduler.add_job(
        engine.check_pyramid,
        CronTrigger(hour=int(ph), minute=int(pm), day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="pyramid_check",
        name="Pyramid check",
    )

    summary_time = schedule.get("daily_summary_time", "16:00")
    sh, sm_val = summary_time.split(":")
    scheduler.add_job(
        engine.daily_summary,
        CronTrigger(hour=int(sh), minute=int(sm_val), day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="daily_summary",
        name="Daily summary",
    )

    # -- Periodic reports --
    weekly_time = schedule.get("weekly_report_time", "10:00")
    wh, wm = weekly_time.split(":")
    weekly_day = schedule.get("weekly_report_day", "sat")
    scheduler.add_job(
        engine.weekly_report,
        CronTrigger(hour=int(wh), minute=int(wm), day_of_week=weekly_day, timezone="Asia/Kolkata"),
        id="weekly_report",
        name="Weekly report",
    )

    monthly_time = schedule.get("monthly_report_time", "10:00")
    mh, mm = monthly_time.split(":")
    monthly_day = schedule.get("monthly_report_day", 1)
    scheduler.add_job(
        engine.monthly_report,
        CronTrigger(hour=int(mh), minute=int(mm), day=int(monthly_day), timezone="Asia/Kolkata"),
        id="monthly_report",
        name="Monthly report",
    )

    yearly_time = schedule.get("yearly_report_time", "10:00")
    yh, ym = yearly_time.split(":")
    yearly_month = schedule.get("yearly_report_month", 1)
    yearly_day = schedule.get("yearly_report_day", 1)
    scheduler.add_job(
        engine.yearly_report,
        CronTrigger(
            hour=int(yh), minute=int(ym),
            month=int(yearly_month), day=int(yearly_day),
            timezone="Asia/Kolkata",
        ),
        id="yearly_report",
        name="Yearly report",
    )

    logger.info("Scheduler configured with %d jobs", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  Job: %s (%s)", job.name, job.trigger)

    def _shutdown(signum, frame):
        logger.info("Shutdown signal received (signal %d), saving state...", signum)
        engine._save_snapshot()
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal_mod.signal(signal_mod.SIGINT, _shutdown)
    signal_mod.signal(signal_mod.SIGTERM, _shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        engine._save_snapshot()
        scheduler.shutdown(wait=False)


@app.command()
def rebalance():
    """Run a one-off rebalance now (bypasses last-trading-day calendar guard)."""
    _setup_logging()
    from engine import TradingEngine
    engine = TradingEngine()
    engine.force_rebalance()
    typer.echo("Rebalance complete.")


@app.command()
def status():
    """Print current positions and portfolio state."""
    _setup_logging()
    from engine import TradingEngine
    engine = TradingEngine()
    st = engine.get_status()

    typer.echo(f"\nPortfolio Value:  ₹{st['portfolio_value']:,.0f}")
    typer.echo(f"Portfolio Peak:   ₹{st['portfolio_peak']:,.0f}")
    typer.echo(f"Exposure:         {st['exposure']:.1%}")
    typer.echo(f"Positions:        {st['positions_count']}")
    typer.echo(f"Circuit Breaker:  {'ACTIVE' if st['circuit_breaker_active'] else 'OFF'}")

    if st["positions"]:
        typer.echo(f"\n{'Ticker':<14} {'Weight':>8} {'Entry':>10} {'Stop':>10} {'Sector':<12}")
        typer.echo("-" * 58)
        for p in st["positions"]:
            stop_str = f"₹{p['stop_price']:,.0f}" if p["stop_price"] else "N/A"
            typer.echo(
                f"{p['ticker']:<14} {p['weight']:>7.1%} "
                f"₹{p['entry_price']:>9,.0f} {stop_str:>10} {p['sector']:<12}"
            )


@app.command()
def stops():
    """Show all current stop levels."""
    _setup_logging()
    from engine import TradingEngine
    engine = TradingEngine()
    stop_list = engine.get_stops()

    if not stop_list:
        typer.echo("No open positions.")
        return

    typer.echo(f"\n{'Ticker':<14} {'Entry':>10} {'Stop':>10} {'HWM':>10} {'% from Stop':>12}")
    typer.echo("-" * 60)
    for s in stop_list:
        stop_str = f"₹{s['stop_price']:,.0f}" if s["stop_price"] else "N/A"
        pct_str = f"{s['pct_from_stop']:.1f}%" if s["pct_from_stop"] is not None else "N/A"
        typer.echo(
            f"{s['ticker']:<14} ₹{s['entry_price']:>9,.0f} "
            f"{stop_str:>10} ₹{s['high_watermark']:>9,.0f} {pct_str:>12}"
        )


@app.command()
def regime():
    """Show current regime assessment."""
    _setup_logging()
    from data_manager import fetch_historical_bulk
    from signal_generator import assess_regime
    from constants import BROAD_UNIVERSE

    cfg = _load_config()
    bench_sym = cfg.get("strategy", {}).get("benchmark", "NIFTY 200").replace("NSE:", "")

    typer.echo("Fetching data for regime assessment...")
    data = fetch_historical_bulk(BROAD_UNIVERSE[:20] + [bench_sym], days=400)
    if not data or data["close"].empty or bench_sym not in data["close"].columns:
        typer.echo("Could not fetch benchmark data.")
        return

    bench_close = data["close"][bench_sym].dropna()
    stocks_close = data["close"].drop(columns=[bench_sym], errors="ignore")

    r = assess_regime(bench_close, stocks_close, cfg)

    typer.echo(f"\nRegime:         {r.level.value}")
    typer.echo(f"Allocation:     {r.allocation_pct:.0%}")
    typer.echo(f"Breadth:        {r.breadth:.1%}")
    typer.echo(f"Bench Close:    ₹{r.bench_close:,.0f}")
    typer.echo(f"Bench 50DMA:    ₹{r.bench_50dma:,.0f}")
    typer.echo(f"Bench 200DMA:   ₹{r.bench_200dma:,.0f}")
    typer.echo(f"Bench 3M Ret:   {r.bench_3m_return:.1%}")


@app.command()
def history(limit: int = 20):
    """Show recent trades."""
    _setup_logging()
    from state_manager import StateManager
    sm = StateManager()
    trades = sm.get_recent_trades(limit)

    if not trades:
        typer.echo("No trades recorded.")
        return

    typer.echo(f"\n{'Date':<20} {'Ticker':<14} {'Action':<8} {'Price':>10} {'Weight':>8} {'Reason'}")
    typer.echo("-" * 80)
    for t in trades:
        typer.echo(
            f"{t['date']:<20} {t['ticker']:<14} {t['action']:<8} "
            f"₹{t['price']:>9,.0f} {t['weight_traded']:>7.1%} {t['reason']}"
        )


@app.command()
def token(request_token: str = typer.Option("", help="Request token from Kite login")):
    """Manual Kite token refresh."""
    _setup_logging()
    from token_manager import TokenManager
    tm = TokenManager()

    if request_token:
        if tm.exchange_request_token(request_token):
            typer.echo("Token exchange successful!")
        else:
            typer.echo("Token exchange failed. Check logs.")
    else:
        if tm.validate_token():
            typer.echo("Current token is valid.")
        else:
            typer.echo(f"Token invalid/expired. Login at:\n  {tm.get_login_url()}")
            typer.echo("Then run: python main.py token --request-token YOUR_TOKEN")


@app.command()
def report(
    period: str = typer.Argument(..., help="Report period: weekly, monthly, or yearly"),
    date: str = typer.Option("", help="Reference date (YYYY-MM-DD). Defaults to today."),
    send: bool = typer.Option(False, help="Also send the report via Telegram"),
):
    """Generate a trading report in Excel format."""
    _setup_logging()
    from datetime import date as date_cls, datetime as dt_cls
    from report_generator import generate_report
    from state_manager import StateManager

    if period not in ("weekly", "monthly", "yearly"):
        typer.echo("Period must be one of: weekly, monthly, yearly")
        raise typer.Exit(code=1)

    ref_date = date_cls.fromisoformat(date) if date else None
    sm = StateManager()

    typer.echo(f"Generating {period} report...")
    report_path = generate_report(period, reference_date=ref_date, state_manager=sm)
    typer.echo(f"Report saved to: {report_path}")

    if send:
        from notifier import TelegramNotifier
        notifier = TelegramNotifier()
        caption = f"<b>{period.upper()} REPORT</b>"
        if notifier.send_document(report_path, caption):
            typer.echo("Report sent via Telegram.")
        else:
            typer.echo("Telegram send failed (check bot_token / chat_id).")


@app.command()
def backtest():
    """Run the historical backtest using config.yaml settings."""
    _setup_logging()
    typer.echo("Running backtest with config.yaml settings...")
    try:
        from backtest import run_backtest_from_config
        run_backtest_from_config()
    except ImportError:
        typer.echo("backtest.py not found or not configured. See backtest-compat setup.")
    except Exception as exc:
        typer.echo(f"Backtest error: {exc}")


if __name__ == "__main__":
    app()
