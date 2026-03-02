"""Add scenarios v1 conversation snapshot fields.

Revision ID: 034_add_scenarios_v1
Revises: 033_add_ens_v3_loop_memory_compression
Create Date: 2026-03-01
"""

from alembic import op
import sqlalchemy as sa


revision = "034_add_scenarios_v1"
down_revision = "033_add_ens_v3_loop_memory_compression"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "conversations"):
        return

    if not _has_column(inspector, "conversations", "scenario_source"):
        op.add_column(
            "conversations",
            sa.Column("scenario_source", sa.String(length=20), nullable=False, server_default="none"),
        )
        op.execute("UPDATE conversations SET scenario_source = 'none' WHERE scenario_source IS NULL")

    inspector = sa.inspect(bind)
    if not _has_column(inspector, "conversations", "scenario_id"):
        op.add_column("conversations", sa.Column("scenario_id", sa.String(length=36), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_column(inspector, "conversations", "scenario_title"):
        op.add_column("conversations", sa.Column("scenario_title", sa.String(length=200), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_column(inspector, "conversations", "scenario_text"):
        op.add_column("conversations", sa.Column("scenario_text", sa.Text(), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_column(inspector, "conversations", "scenario_settings_json"):
        op.add_column("conversations", sa.Column("scenario_settings_json", sa.JSON(), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "conversations", "ix_conversations_scenario_id"):
        op.create_index("ix_conversations_scenario_id", "conversations", ["scenario_id"])


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "conversations"):
        return

    if _has_index(inspector, "conversations", "ix_conversations_scenario_id"):
        op.drop_index("ix_conversations_scenario_id", table_name="conversations")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "conversations", "scenario_settings_json"):
        op.drop_column("conversations", "scenario_settings_json")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "conversations", "scenario_text"):
        op.drop_column("conversations", "scenario_text")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "conversations", "scenario_title"):
        op.drop_column("conversations", "scenario_title")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "conversations", "scenario_id"):
        op.drop_column("conversations", "scenario_id")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "conversations", "scenario_source"):
        op.drop_column("conversations", "scenario_source")
