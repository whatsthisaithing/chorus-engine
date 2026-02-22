"""Add ENS v3 scheduler hardening fields and constraints.

Revision ID: 027_add_ens_v3_scheduler_hardening
Revises: 026_add_ens_v3_scheduler_tables
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "027_add_ens_v3_scheduler_hardening"
down_revision = "026_add_ens_v3_scheduler_tables"
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

    if _has_table(inspector, "ens_signal_queue"):
        if not _has_column(inspector, "ens_signal_queue", "claimed_at_us"):
            op.add_column("ens_signal_queue", sa.Column("claimed_at_us", sa.Integer(), nullable=True))

        inspector = sa.inspect(bind)
        if not _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_claimed_at_us"):
            op.create_index(
                "ix_ens_signal_queue_claimed_at_us",
                "ens_signal_queue",
                ["claimed_at_us"],
            )
        if not _has_index(inspector, "ens_signal_queue", "uq_ens_signal_queue_idempotency_key"):
            op.create_index(
                "uq_ens_signal_queue_idempotency_key",
                "ens_signal_queue",
                ["idempotency_key"],
                unique=True,
            )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ens_signal_queue"):
        if _has_index(inspector, "ens_signal_queue", "uq_ens_signal_queue_idempotency_key"):
            op.drop_index("uq_ens_signal_queue_idempotency_key", table_name="ens_signal_queue")
        if _has_index(inspector, "ens_signal_queue", "ix_ens_signal_queue_claimed_at_us"):
            op.drop_index("ix_ens_signal_queue_claimed_at_us", table_name="ens_signal_queue")

        inspector = sa.inspect(bind)
        if _has_column(inspector, "ens_signal_queue", "claimed_at_us"):
            op.drop_column("ens_signal_queue", "claimed_at_us")
