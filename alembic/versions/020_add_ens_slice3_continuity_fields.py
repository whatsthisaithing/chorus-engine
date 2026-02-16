"""Add ENS Slice 3 continuity ownership fields.

Revision ID: 020_add_ens_slice3_continuity_fields
Revises: 019_add_ens_tool_call_requests
Create Date: 2026-02-16
"""

from alembic import op
import sqlalchemy as sa


revision = "020_add_ens_slice3_continuity_fields"
down_revision = "019_add_ens_tool_call_requests"
branch_labels = None
depends_on = None


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def _has_fk(inspector, table_name: str, fk_name: str) -> bool:
    return any(fk.get("name") == fk_name for fk in inspector.get_foreign_keys(table_name))


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)

    # conversation summary provenance + pointer
    if not _has_column(inspector, "conversations", "current_summary_id"):
        op.add_column("conversations", sa.Column("current_summary_id", sa.String(length=36), nullable=True))
    if not _has_index(inspector, "conversations", "ix_conversations_current_summary_id"):
        op.create_index("ix_conversations_current_summary_id", "conversations", ["current_summary_id"])
    # SQLite cannot ALTER TABLE to add FK constraints after creation.
    if dialect != "sqlite" and not _has_fk(inspector, "conversations", "fk_conversations_current_summary_id"):
        op.create_foreign_key(
            "fk_conversations_current_summary_id",
            "conversations",
            "conversation_summaries",
            ["current_summary_id"],
            ["id"],
        )

    if not _has_column(inspector, "conversation_summaries", "range_start_message_id"):
        op.add_column("conversation_summaries", sa.Column("range_start_message_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "conversation_summaries", "range_end_message_id"):
        op.add_column("conversation_summaries", sa.Column("range_end_message_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "conversation_summaries", "analysis_version"):
        op.add_column("conversation_summaries", sa.Column("analysis_version", sa.String(length=50), nullable=True))
    if not _has_column(inspector, "conversation_summaries", "extractor_version"):
        op.add_column("conversation_summaries", sa.Column("extractor_version", sa.String(length=50), nullable=True))
    if not _has_column(inspector, "conversation_summaries", "summary_input_hash"):
        op.add_column("conversation_summaries", sa.Column("summary_input_hash", sa.String(length=64), nullable=True))
    if not _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_range_start_message_id"):
        op.create_index("ix_conversation_summaries_range_start_message_id", "conversation_summaries", ["range_start_message_id"])
    if not _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_range_end_message_id"):
        op.create_index("ix_conversation_summaries_range_end_message_id", "conversation_summaries", ["range_end_message_id"])
    if not _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_analysis_version"):
        op.create_index("ix_conversation_summaries_analysis_version", "conversation_summaries", ["analysis_version"])
    if not _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_summary_input_hash"):
        op.create_index("ix_conversation_summaries_summary_input_hash", "conversation_summaries", ["summary_input_hash"])

    # memory idempotency/provenance fields
    if not _has_column(inspector, "memories", "client_memory_id"):
        op.add_column("memories", sa.Column("client_memory_id", sa.String(length=100), nullable=True))
    if not _has_column(inspector, "memories", "source_fingerprint"):
        op.add_column("memories", sa.Column("source_fingerprint", sa.String(length=128), nullable=True))
    if not _has_column(inspector, "memories", "source_kind"):
        op.add_column("memories", sa.Column("source_kind", sa.String(length=40), nullable=True))
    if not _has_index(inspector, "memories", "ix_memories_client_memory_id"):
        op.create_index("ix_memories_client_memory_id", "memories", ["client_memory_id"])
    if not _has_index(inspector, "memories", "ix_memories_source_fingerprint"):
        op.create_index("ix_memories_source_fingerprint", "memories", ["source_fingerprint"])
    if not _has_index(inspector, "memories", "ix_memories_source_kind"):
        op.create_index("ix_memories_source_kind", "memories", ["source_kind"])
    # Use a unique index instead of ALTER TABLE ADD CONSTRAINT for SQLite compatibility.
    if not _has_index(inspector, "memories", "uq_memories_conversation_client_memory_id"):
        op.create_index(
            "uq_memories_conversation_client_memory_id",
            "memories",
            ["conversation_id", "client_memory_id"],
            unique=True,
        )
    if not _has_index(inspector, "memories", "ix_memories_vision_idempotency"):
        op.create_index(
            "ix_memories_vision_idempotency",
            "memories",
            ["conversation_id", "thread_id", "source_kind", "source_fingerprint"],
            unique=True,
        )

    # moment pin provenance fields
    if not _has_column(inspector, "moment_pins", "extractor_version"):
        op.add_column("moment_pins", sa.Column("extractor_version", sa.String(length=50), nullable=True))
    if not _has_column(inspector, "moment_pins", "selection_fingerprint"):
        op.add_column("moment_pins", sa.Column("selection_fingerprint", sa.String(length=128), nullable=True))
    if not _has_index(inspector, "moment_pins", "ix_moment_pins_selection_fingerprint"):
        op.create_index("ix_moment_pins_selection_fingerprint", "moment_pins", ["selection_fingerprint"])


def downgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name
    inspector = sa.inspect(bind)

    if _has_index(inspector, "moment_pins", "ix_moment_pins_selection_fingerprint"):
        op.drop_index("ix_moment_pins_selection_fingerprint", table_name="moment_pins")
    if _has_column(inspector, "moment_pins", "selection_fingerprint"):
        op.drop_column("moment_pins", "selection_fingerprint")
    if _has_column(inspector, "moment_pins", "extractor_version"):
        op.drop_column("moment_pins", "extractor_version")

    if _has_index(inspector, "memories", "ix_memories_vision_idempotency"):
        op.drop_index("ix_memories_vision_idempotency", table_name="memories")
    if _has_index(inspector, "memories", "uq_memories_conversation_client_memory_id"):
        op.drop_index("uq_memories_conversation_client_memory_id", table_name="memories")
    if _has_index(inspector, "memories", "ix_memories_source_kind"):
        op.drop_index("ix_memories_source_kind", table_name="memories")
    if _has_index(inspector, "memories", "ix_memories_source_fingerprint"):
        op.drop_index("ix_memories_source_fingerprint", table_name="memories")
    if _has_index(inspector, "memories", "ix_memories_client_memory_id"):
        op.drop_index("ix_memories_client_memory_id", table_name="memories")
    if _has_column(inspector, "memories", "source_kind"):
        op.drop_column("memories", "source_kind")
    if _has_column(inspector, "memories", "source_fingerprint"):
        op.drop_column("memories", "source_fingerprint")
    if _has_column(inspector, "memories", "client_memory_id"):
        op.drop_column("memories", "client_memory_id")

    if _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_summary_input_hash"):
        op.drop_index("ix_conversation_summaries_summary_input_hash", table_name="conversation_summaries")
    if _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_analysis_version"):
        op.drop_index("ix_conversation_summaries_analysis_version", table_name="conversation_summaries")
    if _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_range_end_message_id"):
        op.drop_index("ix_conversation_summaries_range_end_message_id", table_name="conversation_summaries")
    if _has_index(inspector, "conversation_summaries", "ix_conversation_summaries_range_start_message_id"):
        op.drop_index("ix_conversation_summaries_range_start_message_id", table_name="conversation_summaries")
    if _has_column(inspector, "conversation_summaries", "summary_input_hash"):
        op.drop_column("conversation_summaries", "summary_input_hash")
    if _has_column(inspector, "conversation_summaries", "extractor_version"):
        op.drop_column("conversation_summaries", "extractor_version")
    if _has_column(inspector, "conversation_summaries", "analysis_version"):
        op.drop_column("conversation_summaries", "analysis_version")
    if _has_column(inspector, "conversation_summaries", "range_end_message_id"):
        op.drop_column("conversation_summaries", "range_end_message_id")
    if _has_column(inspector, "conversation_summaries", "range_start_message_id"):
        op.drop_column("conversation_summaries", "range_start_message_id")

    if dialect != "sqlite" and _has_fk(inspector, "conversations", "fk_conversations_current_summary_id"):
        op.drop_constraint("fk_conversations_current_summary_id", "conversations", type_="foreignkey")
    if _has_index(inspector, "conversations", "ix_conversations_current_summary_id"):
        op.drop_index("ix_conversations_current_summary_id", table_name="conversations")
    if _has_column(inspector, "conversations", "current_summary_id"):
        op.drop_column("conversations", "current_summary_id")
