"""Initial migration: apply existing backend_schema.sql

This migration executes the raw SQL file `backend_schema.sql` which contains
the canonical DDL for the project's database schema.
"""
from alembic import op
import sqlalchemy as sa
import os

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    sql_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'backend_schema.sql')
    # Resolve through realpath
    sql_path = os.path.realpath(sql_path)
    with open(sql_path, 'r') as f:
        sql = f.read()
    conn = op.get_bind()
    for stmt in [s for s in sql.split(';') if s.strip()]:
        conn.execute(sa.text(stmt))


def downgrade() -> None:
    # No-op: dropping all tables is destructive. Manual rollback required.
    pass
