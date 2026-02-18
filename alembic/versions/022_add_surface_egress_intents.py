"""Add ENS Slice 6.5 surface egress intent outbox.

Revision ID: 022_add_surface_egress_intents
Revises: 021_add_surface_bindings
Create Date: 2026-02-18
"""

from alembic import op
import sqlalchemy as sa


revision = "022_add_surface_egress_intents"
down_revision = "021_add_surface_bindings"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "surface_egress_intents"):
        op.create_table(
            "surface_egress_intents",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("surface_id", sa.String(length=20), nullable=False),
            sa.Column("surface_instance_id", sa.String(length=100), nullable=False, server_default=""),
            sa.Column("external_thread_id", sa.String(length=255), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=True),
            sa.Column("thread_id", sa.String(length=36), nullable=True),
            sa.Column("in_reply_to_message_id", sa.String(length=36), nullable=True),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("next_attempt_at", sa.DateTime(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("idempotency_key", sa.String(length=255), nullable=False),
            sa.Column("trace_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    indexes = (
        ("ix_surface_egress_intents_surface_id", ["surface_id"], False),
        ("ix_surface_egress_intents_surface_instance_id", ["surface_instance_id"], False),
        ("ix_surface_egress_intents_external_thread_id", ["external_thread_id"], False),
        ("ix_surface_egress_intents_relationship_id", ["relationship_id"], False),
        ("ix_surface_egress_intents_conversation_id", ["conversation_id"], False),
        ("ix_surface_egress_intents_thread_id", ["thread_id"], False),
        ("ix_surface_egress_intents_in_reply_to_message_id", ["in_reply_to_message_id"], False),
        ("ix_surface_egress_intents_status", ["status"], False),
        ("ix_surface_egress_intents_idempotency_key", ["idempotency_key"], False),
        ("ix_surface_egress_intents_status_surface", ["status", "surface_id"], False),
        ("uq_surface_egress_intents_idempotency_key", ["idempotency_key"], True),
    )
    for index_name, columns, unique in indexes:
        if not _has_index(inspector, "surface_egress_intents", index_name):
            op.create_index(index_name, "surface_egress_intents", columns, unique=unique)
            inspector = sa.inspect(bind)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_table(inspector, "surface_egress_intents"):
        for index_name in (
            "uq_surface_egress_intents_idempotency_key",
            "ix_surface_egress_intents_status_surface",
            "ix_surface_egress_intents_idempotency_key",
            "ix_surface_egress_intents_status",
            "ix_surface_egress_intents_in_reply_to_message_id",
            "ix_surface_egress_intents_thread_id",
            "ix_surface_egress_intents_conversation_id",
            "ix_surface_egress_intents_relationship_id",
            "ix_surface_egress_intents_external_thread_id",
            "ix_surface_egress_intents_surface_instance_id",
            "ix_surface_egress_intents_surface_id",
        ):
            if _has_index(inspector, "surface_egress_intents", index_name):
                op.drop_index(index_name, table_name="surface_egress_intents")
                inspector = sa.inspect(bind)
        op.drop_table("surface_egress_intents")
