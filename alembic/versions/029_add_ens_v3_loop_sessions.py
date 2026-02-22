"""Add ENS v3 loop sessions and loop progression coalescing index.

Revision ID: 029_add_ens_v3_loop_sessions
Revises: 028_add_ens_v3_floor_control_state
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "029_add_ens_v3_loop_sessions"
down_revision = "028_add_ens_v3_floor_control_state"
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
        op.create_table(
            "ens_loop_sessions",
            sa.Column("loop_id", sa.String(length=36), nullable=False),
            sa.Column("loop_kind", sa.String(length=100), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=False),
            sa.Column("conversation_id", sa.String(length=36), nullable=True),
            sa.Column("surface_id", sa.String(length=20), nullable=True),
            sa.Column("step_index", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("step_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("token_budget_used", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tool_budget_used", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("state", sa.String(length=30), nullable=False, server_default="running"),
            sa.Column("stop_reason", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.PrimaryKeyConstraint("loop_id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_loop_kind"):
        op.create_index("ix_ens_loop_sessions_loop_kind", "ens_loop_sessions", ["loop_kind"])
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_relationship_id"):
        op.create_index("ix_ens_loop_sessions_relationship_id", "ens_loop_sessions", ["relationship_id"])
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_conversation_id"):
        op.create_index("ix_ens_loop_sessions_conversation_id", "ens_loop_sessions", ["conversation_id"])
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_surface_id"):
        op.create_index("ix_ens_loop_sessions_surface_id", "ens_loop_sessions", ["surface_id"])
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_state"):
        op.create_index("ix_ens_loop_sessions_state", "ens_loop_sessions", ["state"])
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_relationship_state"):
        op.create_index(
            "ix_ens_loop_sessions_relationship_state",
            "ens_loop_sessions",
            ["relationship_id", "state"],
        )
    if not _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_surface_state"):
        op.create_index(
            "ix_ens_loop_sessions_surface_state",
            "ens_loop_sessions",
            ["surface_id", "state"],
        )

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_signal_queue"):
        if not _has_column(inspector, "ens_signal_queue", "loop_id"):
            op.add_column("ens_signal_queue", sa.Column("loop_id", sa.String(length=36), nullable=True))

        inspector = sa.inspect(bind)
        if not _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_loop_id"):
            op.create_index("ix_ens_signal_queue_loop_id", "ens_signal_queue", ["loop_id"])
        if not _has_index(inspector, "ens_signal_queue", "uq_ens_signal_queue_pending_loop_progression_by_loop"):
            op.create_index(
                "uq_ens_signal_queue_pending_loop_progression_by_loop",
                "ens_signal_queue",
                ["loop_id"],
                unique=True,
                sqlite_where=sa.text(
                    "signal_type = 'loop_progression' AND status = 'pending' AND loop_id IS NOT NULL"
                ),
            )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ens_signal_queue"):
        if _has_index(inspector, "ens_signal_queue", "uq_ens_signal_queue_pending_loop_progression_by_loop"):
            op.drop_index("uq_ens_signal_queue_pending_loop_progression_by_loop", table_name="ens_signal_queue")
        if _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_loop_id"):
            op.drop_index("ix_ens_signal_queue_loop_id", table_name="ens_signal_queue")

        inspector = sa.inspect(bind)
        if _has_column(inspector, "ens_signal_queue", "loop_id"):
            op.drop_column("ens_signal_queue", "loop_id")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_sessions"):
        for idx in [
            "ix_ens_loop_sessions_surface_state",
            "ix_ens_loop_sessions_relationship_state",
            "ix_ens_loop_sessions_state",
            "ix_ens_loop_sessions_surface_id",
            "ix_ens_loop_sessions_conversation_id",
            "ix_ens_loop_sessions_relationship_id",
            "ix_ens_loop_sessions_loop_kind",
        ]:
            if _has_index(inspector, "ens_loop_sessions", idx):
                op.drop_index(idx, table_name="ens_loop_sessions")
        op.drop_table("ens_loop_sessions")
