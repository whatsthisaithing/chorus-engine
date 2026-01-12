"""Alembic migration environment configuration.

This module configures the migration environment for Chorus Engine's database.
It imports all models to ensure SQLAlchemy's metadata is fully populated.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import the Base model and all database models to populate metadata
from chorus_engine.db.database import Base, engine as chorus_engine

# Import all model files to ensure they're registered with Base
import chorus_engine.models.conversation
import chorus_engine.models.workflow
import chorus_engine.models.document
import chorus_engine.models.custom_model
import chorus_engine.repositories.conversation_repository
import chorus_engine.repositories.message_repository
import chorus_engine.repositories.memory_repository
import chorus_engine.repositories.thread_repository
import chorus_engine.repositories.image_repository
import chorus_engine.repositories.audio_repository
import chorus_engine.repositories.voice_sample_repository
import chorus_engine.repositories.workflow_repository

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # Use the engine from the MigrationManager if available
        # This is set programmatically when migrations are run from the app
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
