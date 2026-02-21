"""Add branching provenance fields for relationship v1.1.

Revision ID: 025_add_branching_provenance_fields
Revises: 024_add_conversation_segments
Create Date: 2026-02-21
"""

from alembic import op
import sqlalchemy as sa


revision = "025_add_branching_provenance_fields"
down_revision = "024_add_conversation_segments"
branch_labels = None
depends_on = None


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Conversation provenance and recap marker
    if not _has_column(inspector, "conversations", "origin_conversation_id"):
        op.add_column("conversations", sa.Column("origin_conversation_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "conversations", "origin_mode"):
        op.add_column("conversations", sa.Column("origin_mode", sa.String(length=20), nullable=True))
    if not _has_column(inspector, "conversations", "origin_segment_id"):
        op.add_column("conversations", sa.Column("origin_segment_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "conversations", "origin_segment_ids_json"):
        op.add_column("conversations", sa.Column("origin_segment_ids_json", sa.JSON(), nullable=True))
    if not _has_column(inspector, "conversations", "branch_created_at"):
        op.add_column("conversations", sa.Column("branch_created_at", sa.DateTime(), nullable=True))
    if not _has_column(inspector, "conversations", "branch_origin_recap_injected_at"):
        op.add_column(
            "conversations",
            sa.Column("branch_origin_recap_injected_at", sa.DateTime(), nullable=True),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "conversations", "ix_conversations_origin_conversation_id"):
        op.create_index("ix_conversations_origin_conversation_id", "conversations", ["origin_conversation_id"])
    if not _has_index(inspector, "conversations", "ix_conversations_origin_mode"):
        op.create_index("ix_conversations_origin_mode", "conversations", ["origin_mode"])
    if not _has_index(inspector, "conversations", "ix_conversations_origin_segment_id"):
        op.create_index("ix_conversations_origin_segment_id", "conversations", ["origin_segment_id"])
    if not _has_index(inspector, "conversations", "ix_conversations_branch_created_at"):
        op.create_index("ix_conversations_branch_created_at", "conversations", ["branch_created_at"])

    # Imported-message provenance
    if not _has_column(inspector, "messages", "imported_from_conversation_id"):
        op.add_column("messages", sa.Column("imported_from_conversation_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "messages", "imported_from_message_id"):
        op.add_column("messages", sa.Column("imported_from_message_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "messages", "imported_from_segment_id"):
        op.add_column("messages", sa.Column("imported_from_segment_id", sa.String(length=36), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "messages", "ix_messages_imported_from_conversation_id"):
        op.create_index(
            "ix_messages_imported_from_conversation_id",
            "messages",
            ["imported_from_conversation_id"],
        )
    if not _has_index(inspector, "messages", "ix_messages_imported_from_message_id"):
        op.create_index(
            "ix_messages_imported_from_message_id",
            "messages",
            ["imported_from_message_id"],
        )
    if not _has_index(inspector, "messages", "ix_messages_imported_from_segment_id"):
        op.create_index(
            "ix_messages_imported_from_segment_id",
            "messages",
            ["imported_from_segment_id"],
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_index(inspector, "messages", "ix_messages_imported_from_segment_id"):
        op.drop_index("ix_messages_imported_from_segment_id", table_name="messages")
    if _has_index(inspector, "messages", "ix_messages_imported_from_message_id"):
        op.drop_index("ix_messages_imported_from_message_id", table_name="messages")
    if _has_index(inspector, "messages", "ix_messages_imported_from_conversation_id"):
        op.drop_index("ix_messages_imported_from_conversation_id", table_name="messages")
    if _has_column(inspector, "messages", "imported_from_segment_id"):
        op.drop_column("messages", "imported_from_segment_id")
    if _has_column(inspector, "messages", "imported_from_message_id"):
        op.drop_column("messages", "imported_from_message_id")
    if _has_column(inspector, "messages", "imported_from_conversation_id"):
        op.drop_column("messages", "imported_from_conversation_id")

    inspector = sa.inspect(bind)
    if _has_index(inspector, "conversations", "ix_conversations_branch_created_at"):
        op.drop_index("ix_conversations_branch_created_at", table_name="conversations")
    if _has_index(inspector, "conversations", "ix_conversations_origin_segment_id"):
        op.drop_index("ix_conversations_origin_segment_id", table_name="conversations")
    if _has_index(inspector, "conversations", "ix_conversations_origin_mode"):
        op.drop_index("ix_conversations_origin_mode", table_name="conversations")
    if _has_index(inspector, "conversations", "ix_conversations_origin_conversation_id"):
        op.drop_index("ix_conversations_origin_conversation_id", table_name="conversations")
    if _has_column(inspector, "conversations", "branch_origin_recap_injected_at"):
        op.drop_column("conversations", "branch_origin_recap_injected_at")
    if _has_column(inspector, "conversations", "branch_created_at"):
        op.drop_column("conversations", "branch_created_at")
    if _has_column(inspector, "conversations", "origin_segment_ids_json"):
        op.drop_column("conversations", "origin_segment_ids_json")
    if _has_column(inspector, "conversations", "origin_segment_id"):
        op.drop_column("conversations", "origin_segment_id")
    if _has_column(inspector, "conversations", "origin_mode"):
        op.drop_column("conversations", "origin_mode")
    if _has_column(inspector, "conversations", "origin_conversation_id"):
        op.drop_column("conversations", "origin_conversation_id")
