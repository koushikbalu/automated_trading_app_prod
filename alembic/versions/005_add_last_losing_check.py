"""Add last_losing_check column to positions table.

Revision ID: 005
Revises: 004
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("positions", sa.Column("last_losing_check", sa.Date, nullable=True))


def downgrade() -> None:
    op.drop_column("positions", "last_losing_check")
