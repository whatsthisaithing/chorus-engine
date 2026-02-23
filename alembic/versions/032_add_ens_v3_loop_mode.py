"""Add loop_mode to ENS loop sessions.

Revision ID: 032_add_ens_v3_loop_mode
Revises: 031_add_ens_v3_loop_step_event_uniqueness
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "032_add_ens_v3_loop_mode"
down_revision = "031_add_ens_v3_loop_step_event_uniqueness"
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
    if not _has_table(inspector, "ens_loop_sessions"):
        return
    if not _has_column(inspector, "ens_loop_sessions", "loop_mode"):
        op.add_column(
            "ens_loop_sessions",
            sa.Column("loop_mode", sa.String(length=20), nullable=False, server_default="visible"),
        )
    bind.execute(sa.text("UPDATE ens_loop_sessions SET loop_mode='visible' WHERE loop_mode IS NULL OR loop_mode = ''"))
    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_loop_mode"):
        op.create_index("ix_ens_loop_sessions_loop_mode", "ens_loop_sessions", ["loop_mode"])


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_sessions"):
        if _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_loop_mode"):
            op.drop_index("ix_ens_loop_sessions_loop_mode", table_name="ens_loop_sessions")
        inspector = sa.inspect(bind)
        if _has_column(inspector, "ens_loop_sessions", "loop_mode"):
            op.drop_column("ens_loop_sessions", "loop_mode")

