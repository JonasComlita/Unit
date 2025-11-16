"""Merge migration for multiple 0001 heads

Revision ID: 0002_merge_0001_heads
Revises: 0001_initial, 0001_partition_games
Create Date: 2025-11-16 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_merge_0001_heads'
down_revision = ('0001_initial', '0001_partition_games')
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge point for parallel 0001 migrations.

    This migration is intentionally a no-op that unifies two independent
    migration branches into a single linear history so `alembic upgrade head`
    can be used. If further migration steps are needed to reconcile schema
    differences, implement them here.
    """
    pass


def downgrade() -> None:
    # Downgrading a merge point is non-trivial; no-op provided.
    pass
