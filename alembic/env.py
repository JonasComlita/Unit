import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

# set sqlalchemy.url from env if provided
db_url = os.getenv('DATABASE_URL')
if db_url:
    config.set_main_option('sqlalchemy.url', db_url)


def run_migrations_offline():
    url = config.get_main_option('sqlalchemy.url')
    context.configure(url=url, target_metadata=None, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    try:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix='sqlalchemy.',
            poolclass=pool.NullPool,
        )
    except Exception:
        # Some alembic.ini files use interpolation like '%(DATABASE_URL)s' which
        # configparser may try to expand before env vars are applied and raise
        # InterpolationMissingOptionError. Try to recover by resolving the
        # placeholder from environment variables or by using DATABASE_URL.
        import re
        raw_url = None
        try:
            raw_url = config.get_main_option('sqlalchemy.url')
        except Exception:
            raw_url = None

        resolved = None
        if raw_url and '%(' in raw_url:
            # Replace %(FOO)s occurrences with environment variables if present
            def _repl(m):
                key = m.group(1)
                return os.getenv(key) or os.getenv(key.lower()) or ''
            resolved = re.sub(r"%\(([^)]+)\)s", _repl, raw_url)

        if not resolved:
            resolved = os.getenv('DATABASE_URL')

        if not resolved:
            # Give a clearer error explaining what to do next
            raise RuntimeError(
                "alembic config interpolation failed: sqlalchemy.url contains placeholders "
                "and DATABASE_URL is not set. Set the DATABASE_URL env var or update alembic.ini."
            )

        connectable = engine_from_config(
            {'sqlalchemy.url': resolved},
            prefix='sqlalchemy.',
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
