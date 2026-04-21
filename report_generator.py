"""Periodic report generation from live trading database.

Queries RDS tables (trades, portfolio_state, regime_history,
rebalance_history) and produces formatted Excel reports for
weekly, monthly, and yearly periods.
"""

from __future__ import annotations

import json
import logging
import tempfile
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from state_manager import (
    PortfolioStateRow,
    RebalanceHistoryRow,
    RegimeHistoryRow,
    TradeRow,
    StateManager,
)
from utils import annualized_return, annualized_vol, max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Period bounds
# ---------------------------------------------------------------------------

def get_period_bounds(
    period: str, reference_date: date | None = None,
) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes for the *previous* completed period.

    ``reference_date`` defaults to today.  The returned range is inclusive
    on both sides at the date level (times are 00:00:00 and 23:59:59).
    """
    ref = reference_date or date.today()

    if period == "weekly":
        end_date = ref - timedelta(days=ref.weekday() + 2)  # last Friday
        start_date = end_date - timedelta(days=4)            # Monday of that week
    elif period == "monthly":
        first_of_this_month = ref.replace(day=1)
        last_of_prev = first_of_this_month - timedelta(days=1)
        start_date = last_of_prev.replace(day=1)
        end_date = last_of_prev
    elif period == "yearly":
        start_date = date(ref.year - 1, 1, 1)
        end_date = date(ref.year - 1, 12, 31)
    else:
        raise ValueError(f"Unknown period: {period!r} (use weekly/monthly/yearly)")

    return (
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time()),
    )


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _query_period_data(
    session: Session, start: datetime, end: datetime,
) -> dict[str, pd.DataFrame]:
    """Pull all relevant rows for [start, end] from the four main tables."""

    snapshots = session.query(PortfolioStateRow).filter(
        PortfolioStateRow.date >= start, PortfolioStateRow.date <= end,
    ).order_by(PortfolioStateRow.date).all()

    trades = session.query(TradeRow).filter(
        TradeRow.date >= start, TradeRow.date <= end,
    ).order_by(TradeRow.date).all()

    regimes = session.query(RegimeHistoryRow).filter(
        RegimeHistoryRow.date >= start, RegimeHistoryRow.date <= end,
    ).order_by(RegimeHistoryRow.date).all()

    rebalances = session.query(RebalanceHistoryRow).filter(
        RebalanceHistoryRow.rebalance_date >= start,
        RebalanceHistoryRow.rebalance_date <= end,
    ).order_by(RebalanceHistoryRow.rebalance_date).all()

    snap_df = pd.DataFrame([{
        "date": r.date,
        "portfolio_value": r.portfolio_value,
        "portfolio_peak": r.portfolio_peak,
        "circuit_breaker_active": r.circuit_breaker_active,
        "cash_weight": r.cash_weight,
        "exposure": r.exposure,
        "positions_count": r.positions_count,
    } for r in snapshots])

    trades_df = pd.DataFrame([{
        "date": r.date,
        "ticker": r.ticker,
        "action": r.action,
        "price": r.price,
        "weight_traded": r.weight_traded,
        "reason": r.reason or "",
        "order_id": r.order_id or "",
        "costs": r.costs,
        "holding_days": r.holding_days,
        "pnl_pct": r.pnl_pct,
    } for r in trades])

    regime_df = pd.DataFrame([{
        "date": r.date,
        "level": r.level,
        "allocation_pct": r.allocation_pct,
        "breadth": r.breadth,
        "bench_close": r.bench_close,
        "bench_50dma": r.bench_50dma,
        "bench_200dma": r.bench_200dma,
        "bench_3m_return": r.bench_3m_return,
    } for r in regimes])

    rebal_df = pd.DataFrame([{
        "rebalance_date": r.rebalance_date,
        "num_selected": r.num_selected,
        "allocation_pct": r.allocation_pct,
        "picks": r.picks_json or "{}",
    } for r in rebalances])

    return {
        "snapshots": snap_df,
        "trades": trades_df,
        "regime": regime_df,
        "rebalances": rebal_df,
    }


# ---------------------------------------------------------------------------
# Roundtrip matching  (same logic as backtest.py)
# ---------------------------------------------------------------------------

def _build_roundtrips(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    open_rt: dict[str, dict] = {}

    for _, row in trades_df.sort_values("date").iterrows():
        ticker = row["ticker"]
        action = row["action"]
        price = row["price"]
        weight = row["weight_traded"]
        dt = row["date"]

        if action == "BUY":
            if ticker not in open_rt:
                open_rt[ticker] = {
                    "total_cost": price * weight,
                    "total_weight": weight,
                    "entry_date": dt,
                    "pyramids": 0,
                }
            else:
                pos = open_rt[ticker]
                pos["total_cost"] += price * weight
                pos["total_weight"] += weight
                pos["pyramids"] += 1
        elif action in ("SELL", "REBAL_EXIT"):
            if ticker in open_rt:
                pos = open_rt.pop(ticker)
                wavg_entry = pos["total_cost"] / pos["total_weight"]
                ret = price / wavg_entry - 1 if wavg_entry > 0 else np.nan
                hold = (pd.Timestamp(dt) - pd.Timestamp(pos["entry_date"])).days
                rows.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": dt,
                    "entry_price": wavg_entry,
                    "exit_price": price,
                    "weight": pos["total_weight"],
                    "return": ret,
                    "holding_days": hold,
                    "pyramids": pos["pyramids"],
                    "exit_reason": row.get("reason", action),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    snap_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    roundtrip_df: pd.DataFrame,
    annual_rf: float = 0.04,
) -> pd.DataFrame:
    """Build a summary DataFrame of key performance metrics."""

    metrics: list[dict] = []

    if snap_df.empty:
        metrics.append({"Metric": "Status", "Value": "No portfolio data for this period"})
        return pd.DataFrame(metrics)

    snap_df = snap_df.sort_values("date").copy()
    start_val = snap_df["portfolio_value"].iloc[0]
    end_val = snap_df["portfolio_value"].iloc[-1]
    period_return = (end_val / start_val - 1) if start_val > 0 else np.nan

    equity = snap_df.set_index("date")["portfolio_value"]
    daily_ret = equity.pct_change().dropna()

    mdd = max_drawdown(equity)
    ann_ret = annualized_return(equity)
    ann_vol = annualized_vol(daily_ret) if len(daily_ret) > 1 else np.nan
    sr = sharpe_ratio(daily_ret, annual_rf) if len(daily_ret) > 1 else np.nan

    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = float(neg_ret.std(ddof=1) * np.sqrt(252)) if len(neg_ret) > 1 else np.nan

    if downside_vol and downside_vol > 0:
        sortino = float((daily_ret - daily_rf).mean() / neg_ret.std(ddof=1) * np.sqrt(252))
    else:
        sortino = np.nan

    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    peak = equity.cummax()
    in_dd = equity < peak
    dd_groups = (~in_dd).cumsum()
    dd_dur = in_dd.groupby(dd_groups).sum()
    longest_dd = int(dd_dur.max()) if len(dd_dur) > 0 else 0

    avg_exposure = snap_df["exposure"].mean()
    pct_in_market = (snap_df["exposure"] > 0).mean()
    avg_positions = snap_df["positions_count"].mean()

    total_trades = len(trades_df) if not trades_df.empty else 0
    total_costs = trades_df["costs"].sum() if not trades_df.empty else 0.0

    if not roundtrip_df.empty:
        rt_ret = roundtrip_df["return"].dropna()
        win_rate = float((rt_ret > 0).mean()) if len(rt_ret) else np.nan
        avg_winner = float(rt_ret[rt_ret > 0].mean()) if (rt_ret > 0).any() else 0.0
        avg_loser = float(rt_ret[rt_ret <= 0].mean()) if (rt_ret <= 0).any() else 0.0
        gross_profit = rt_ret[rt_ret > 0].sum()
        gross_loss = abs(rt_ret[rt_ret <= 0].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        avg_hold = float(roundtrip_df["holding_days"].mean())
    else:
        win_rate = avg_winner = avg_loser = profit_factor = avg_hold = np.nan

    metrics = [
        {"Metric": "Period Start", "Value": str(snap_df["date"].iloc[0].date())},
        {"Metric": "Period End", "Value": str(snap_df["date"].iloc[-1].date())},
        {"Metric": "Start Value", "Value": start_val},
        {"Metric": "End Value", "Value": end_val},
        {"Metric": "Period Return", "Value": period_return},
        {"Metric": "Annualized Return", "Value": ann_ret},
        {"Metric": "Max Drawdown", "Value": mdd},
        {"Metric": "Longest Drawdown (days)", "Value": longest_dd},
        {"Metric": "Annualized Volatility", "Value": ann_vol},
        {"Metric": "Downside Volatility", "Value": downside_vol},
        {"Metric": "Sharpe (rf=4%)", "Value": sr},
        {"Metric": "Sortino (rf=4%)", "Value": sortino},
        {"Metric": "Calmar Ratio", "Value": calmar},
        {"Metric": "Win Rate (by trade)", "Value": win_rate},
        {"Metric": "Avg Winner", "Value": avg_winner},
        {"Metric": "Avg Loser", "Value": avg_loser},
        {"Metric": "Profit Factor", "Value": profit_factor},
        {"Metric": "Total Roundtrips", "Value": len(roundtrip_df)},
        {"Metric": "Total Trades", "Value": total_trades},
        {"Metric": "Total Trading Costs", "Value": total_costs},
        {"Metric": "Avg Holding Period (days)", "Value": avg_hold},
        {"Metric": "Average Exposure", "Value": avg_exposure},
        {"Metric": "% Time in Market", "Value": pct_in_market},
        {"Metric": "Average Positions", "Value": avg_positions},
    ]
    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------------
# Daily equity sheet
# ---------------------------------------------------------------------------

def _build_equity_sheet(snap_df: pd.DataFrame) -> pd.DataFrame:
    if snap_df.empty:
        return pd.DataFrame()

    df = snap_df.sort_values("date").copy()
    df["daily_return"] = df["portfolio_value"].pct_change()
    base = df["portfolio_value"].iloc[0]
    df["cumulative_return"] = df["portfolio_value"] / base - 1
    peak = df["portfolio_value"].cummax()
    df["drawdown"] = df["portfolio_value"] / peak - 1
    return df[["date", "portfolio_value", "portfolio_peak", "daily_return",
               "cumulative_return", "drawdown", "exposure", "positions_count"]]


# ---------------------------------------------------------------------------
# Excel writing
# ---------------------------------------------------------------------------

def _write_excel(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Write all report sheets to a formatted .xlsx file."""

    with pd.ExcelWriter(
        output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd",
    ) as writer:
        data["summary"].to_excel(writer, sheet_name="Summary", index=False)
        data["daily_equity"].to_excel(writer, sheet_name="Daily_Equity", index=False)
        data["trades"].to_excel(writer, sheet_name="Trades", index=False)
        if not data["roundtrips"].empty:
            data["roundtrips"].to_excel(writer, sheet_name="Roundtrip_Trades", index=False)
        data["regime"].to_excel(writer, sheet_name="Regime", index=False)
        data["rebalances"].to_excel(writer, sheet_name="Rebalances", index=False)

        workbook = writer.book
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        money_fmt = workbook.add_format({"num_format": "#,##0.00"})
        date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})

        # -- Summary sheet --
        ws = writer.sheets["Summary"]
        ws.set_column("A:A", 28)
        ws.set_column("B:B", 20)

        # -- Daily Equity sheet --
        eq = writer.sheets["Daily_Equity"]
        eq.set_column("A:A", 14, date_fmt)
        eq.set_column("B:C", 18, money_fmt)
        eq.set_column("D:F", 14, pct_fmt)
        eq.set_column("G:G", 12, pct_fmt)
        eq.set_column("H:H", 12)

        # -- Trades sheet --
        if not data["trades"].empty:
            tr = writer.sheets["Trades"]
            tr.set_column("A:A", 14, date_fmt)
            tr.set_column("B:B", 18)
            tr.set_column("C:C", 12)
            tr.set_column("D:D", 14, money_fmt)
            tr.set_column("E:E", 12, pct_fmt)
            tr.set_column("F:F", 28)

        # -- Roundtrip sheet --
        if "Roundtrip_Trades" in writer.sheets:
            rt = writer.sheets["Roundtrip_Trades"]
            rt.set_column("A:A", 18)
            rt.set_column("B:C", 14, date_fmt)
            rt.set_column("D:E", 14, money_fmt)
            rt.set_column("F:F", 12, pct_fmt)
            rt.set_column("G:H", 10)
            rt.set_column("I:I", 12)
            rt.set_column("J:J", 28)

        # -- Regime sheet --
        if not data["regime"].empty:
            rg = writer.sheets["Regime"]
            rg.set_column("A:A", 14, date_fmt)
            rg.set_column("B:B", 14)
            rg.set_column("C:D", 12, pct_fmt)
            rg.set_column("E:H", 14, money_fmt)

        # -- Rebalances sheet --
        if not data["rebalances"].empty:
            rb = writer.sheets["Rebalances"]
            rb.set_column("A:A", 14, date_fmt)
            rb.set_column("B:B", 12)
            rb.set_column("C:C", 12, pct_fmt)
            rb.set_column("D:D", 40)

        # -- Equity chart --
        if not data["daily_equity"].empty and len(data["daily_equity"]) > 1:
            chart = workbook.add_chart({"type": "line"})
            rows = len(data["daily_equity"])
            chart.add_series({
                "name": "Portfolio Value",
                "categories": ["Daily_Equity", 1, 0, rows, 0],
                "values": ["Daily_Equity", 1, 1, rows, 1],
            })
            chart.set_title({"name": "Portfolio Equity Curve"})
            chart.set_x_axis({"name": "Date"})
            chart.set_y_axis({"name": "Portfolio Value (₹)"})
            chart.set_size({"width": 900, "height": 420})
            eq.insert_chart("J2", chart)

    logger.info("Report written: %s", output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    period: str,
    reference_date: date | None = None,
    state_manager: StateManager | None = None,
) -> Path:
    """Generate an Excel report for the given period.

    Returns the Path to the temporary .xlsx file.  Caller is responsible
    for sending / cleaning up the file.
    """
    sm = state_manager or StateManager()
    start, end = get_period_bounds(period, reference_date)

    logger.info(
        "Generating %s report: %s to %s",
        period, start.date(), end.date(),
    )

    with sm._session() as session:
        raw = _query_period_data(session, start, end)

    roundtrips = _build_roundtrips(raw["trades"])
    summary = _compute_metrics(raw["snapshots"], raw["trades"], roundtrips)
    equity = _build_equity_sheet(raw["snapshots"])

    report_data = {
        "summary": summary,
        "daily_equity": equity,
        "trades": raw["trades"],
        "roundtrips": roundtrips,
        "regime": raw["regime"],
        "rebalances": raw["rebalances"],
    }

    label = f"{period}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    output_path = Path(tempfile.gettempdir()) / f"trading_report_{label}.xlsx"
    _write_excel(report_data, output_path)

    return output_path
