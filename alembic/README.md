Alembic migration scaffolding
=============================

This folder is a minimal placeholder for Alembic-based migrations. The project currently
uses `scripts/migrate.sh` to apply the SQL schema in `backend_schema.sql`.

If you want to adopt Alembic:

1. Install alembic in your environment: `pip install alembic`
2. Run: `alembic init alembic` (this will create alembic/ env files)
3. Update `alembic.ini` sqlalchemy.url to point to your DATABASE_URL or set it with env
4. Generate migrations with `alembic revision --autogenerate -m "initial"` and edit as needed
5. Apply with `alembic upgrade head`

For now `scripts/migrate.sh` is the canonical migration path used in CI.
