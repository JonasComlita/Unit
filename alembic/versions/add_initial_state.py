"""Add initial_state column to games table

Revision ID: 001_initial_state
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial_state'
# This migration should apply after the merge point that unifies the two
# independent 0001 heads. Point its down_revision to the merge migration so
# the history is linear when upgrading.
down_revision = '0002_merge_0001_heads'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('games', sa.Column('initial_state', sa.Text, nullable=True))
    op.create_index('idx_games_start_time', 'games', ['start_time'])


def downgrade():
    op.drop_index('idx_games_start_time')
    op.drop_column('games', 'initial_state')