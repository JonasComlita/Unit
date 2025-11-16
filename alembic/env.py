from __future__ import with_statement

import os
from logging.config import fileConfig

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
try:
    fileConfig(config.config_file_name)
except Exception:
    # If the ini logging sections are not present, continue with default logging
    pass

# Provide the database URL via environment variable DATABASE_URL
def get_url():
    return os.environ.get('DATABASE_URL')


def run_migrations_offline():
    url = get_url()
    context.configure(url=url, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = None

    from sqlalchemy import create_engine
    url = get_url()
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(connection=connection)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
