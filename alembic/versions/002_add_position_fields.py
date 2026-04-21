"""Add losing_days and pyramid_count to positions table.

Revision ID: 002
Revises: 001
Create Date: 2026-04-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("positions", sa.Column("losing_days", sa.Integer, nullable=False, server_default="0"))
    op.add_column("positions", sa.Column("pyramid_count", sa.Integer, nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("positions", "pyramid_count")
    op.drop_column("positions", "losing_days")
