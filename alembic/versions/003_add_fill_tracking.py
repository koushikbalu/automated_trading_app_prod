"""Add fill tracking fields to trades and capital_injections_total to portfolio_state.

Revision ID: 003
Revises: 002
Create Date: 2026-04-23
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("filled_qty", sa.Integer, nullable=True))
    op.add_column("trades", sa.Column("requested_qty", sa.Integer, nullable=True))
    op.add_column("trades", sa.Column("fill_status", sa.String(16), nullable=True))
    op.add_column("portfolio_state", sa.Column("capital_injections_total", sa.Float, nullable=True, server_default="0"))


def downgrade() -> None:
    op.drop_column("portfolio_state", "capital_injections_total")
    op.drop_column("trades", "fill_status")
    op.drop_column("trades", "requested_qty")
    op.drop_column("trades", "filled_qty")
