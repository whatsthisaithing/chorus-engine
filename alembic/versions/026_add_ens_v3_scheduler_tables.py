"""Add ENS v3 scheduler tables.

Revision ID: 026_add_ens_v3_scheduler_tables
Revises: 025_add_branching_provenance_fields
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "026_add_ens_v3_scheduler_tables"
down_revision = "025_add_branching_provenance_fields"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "ens_signal_queue"):
        op.create_table(
            "ens_signal_queue",
            sa.Column("queue_id", sa.String(length=36), nullable=False),
            sa.Column("signal_id", sa.String(length=36), nullable=False),
            sa.Column("signal_type", sa.String(length=100), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=True),
            sa.Column("surface_id", sa.String(length=20), nullable=True),
            sa.Column("priority_tier", sa.String(length=20), nullable=False, server_default="system"),
            sa.Column("created_at_us", sa.Integer(), nullable=False),
            sa.Column("idempotency_key", sa.String(length=255), nullable=True),
            sa.Column("signal_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("selected_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.PrimaryKeyConstraint("queue_id"),
            sa.UniqueConstraint("signal_id", name="uq_ens_signal_queue_signal_id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_status_priority_created"):
        op.create_index(
            "ix_ens_signal_queue_status_priority_created",
            "ens_signal_queue",
            ["status", "priority_tier", "created_at_us"],
        )
    if not _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_surface_rel_status"):
        op.create_index(
            "ix_ens_signal_queue_surface_rel_status",
            "ens_signal_queue",
            ["surface_id", "relationship_id", "status"],
        )

    if not _has_table(inspector, "ens_scheduler_ticks"):
        op.create_table(
            "ens_scheduler_ticks",
            sa.Column("tick_id", sa.String(length=36), nullable=False),
            sa.Column("queue_id", sa.String(length=36), nullable=True),
            sa.Column("selected_signal_id", sa.String(length=36), nullable=True),
            sa.Column("reason_trace_json", sa.JSON(), nullable=False),
            sa.Column("tie_break_json", sa.JSON(), nullable=True),
            sa.Column("created_at_us", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.ForeignKeyConstraint(["queue_id"], ["ens_signal_queue.queue_id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("tick_id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_scheduler_ticks", "ix_ens_scheduler_ticks_selected_signal_id"):
        op.create_index(
            "ix_ens_scheduler_ticks_selected_signal_id",
            "ens_scheduler_ticks",
            ["selected_signal_id"],
        )
    if not _has_index(inspector, "ens_scheduler_ticks", "ix_ens_scheduler_ticks_created_at_us"):
        op.create_index(
            "ix_ens_scheduler_ticks_created_at_us",
            "ens_scheduler_ticks",
            ["created_at_us"],
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_index(inspector, "ens_scheduler_ticks", "ix_ens_scheduler_ticks_created_at_us"):
        op.drop_index("ix_ens_scheduler_ticks_created_at_us", table_name="ens_scheduler_ticks")
    if _has_index(inspector, "ens_scheduler_ticks", "ix_ens_scheduler_ticks_selected_signal_id"):
        op.drop_index("ix_ens_scheduler_ticks_selected_signal_id", table_name="ens_scheduler_ticks")
    if _has_table(inspector, "ens_scheduler_ticks"):
        op.drop_table("ens_scheduler_ticks")

    inspector = sa.inspect(bind)
    if _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_surface_rel_status"):
        op.drop_index("ix_ens_signal_queue_surface_rel_status", table_name="ens_signal_queue")
    if _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_status_priority_created"):
        op.drop_index("ix_ens_signal_queue_status_priority_created", table_name="ens_signal_queue")
    if _has_table(inspector, "ens_signal_queue"):
        op.drop_table("ens_signal_queue")

