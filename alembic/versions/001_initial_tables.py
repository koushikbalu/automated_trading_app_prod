"""Initial tables.

Revision ID: 001
Revises: None
Create Date: 2026-04-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(32), nullable=False, unique=True, index=True),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("entry_date", sa.DateTime, nullable=False),
        sa.Column("high_watermark", sa.Float, nullable=False),
        sa.Column("stop_price", sa.Float, nullable=True),
        sa.Column("current_price", sa.Float, nullable=True),
        sa.Column("sector", sa.String(32), nullable=False, server_default="Other"),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.DateTime, nullable=False, index=True),
        sa.Column("ticker", sa.String(32), nullable=False, index=True),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("weight_traded", sa.Float, nullable=False),
        sa.Column("reason", sa.String(128), nullable=True),
        sa.Column("order_id", sa.String(64), nullable=True),
        sa.Column("costs", sa.Float, nullable=False, server_default="0"),
        sa.Column("holding_days", sa.Integer, nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
    )

    op.create_table(
        "portfolio_state",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.DateTime, nullable=False, index=True),
        sa.Column("portfolio_value", sa.Float, nullable=False),
        sa.Column("portfolio_peak", sa.Float, nullable=False),
        sa.Column("circuit_breaker_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("cash_weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("exposure", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("positions_count", sa.Integer, nullable=False, server_default="0"),
    )

    op.create_table(
        "regime_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("date", sa.DateTime, nullable=False, index=True),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("allocation_pct", sa.Float, nullable=False),
        sa.Column("breadth", sa.Float, nullable=False),
        sa.Column("bench_close", sa.Float, nullable=False),
        sa.Column("bench_50dma", sa.Float, nullable=False),
        sa.Column("bench_200dma", sa.Float, nullable=False),
        sa.Column("bench_3m_return", sa.Float, nullable=False),
    )

    op.create_table(
        "rebalance_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("rebalance_date", sa.DateTime, nullable=False, index=True),
        sa.Column("num_selected", sa.Integer, nullable=False),
        sa.Column("allocation_pct", sa.Float, nullable=False),
        sa.Column("picks_json", sa.Text, nullable=True),
    )

    op.create_table(
        "stopped_out_tracker",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("ticker", sa.String(32), nullable=False, index=True),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("stop_date", sa.DateTime, nullable=False),
        sa.Column("month_year", sa.String(7), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("stopped_out_tracker")
    op.drop_table("rebalance_history")
    op.drop_table("regime_history")
    op.drop_table("portfolio_state")
    op.drop_table("trades")
    op.drop_table("positions")
