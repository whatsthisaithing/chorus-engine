"""Add conversation segments for relationship v1.

Revision ID: 024_add_conversation_segments
Revises: 023_relationship_first_v0
Create Date: 2026-02-21
"""

from alembic import op
import sqlalchemy as sa


revision = "024_add_conversation_segments"
down_revision = "023_relationship_first_v0"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "conversation_segments"):
        op.create_table(
            "conversation_segments",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("conversation_id", sa.String(length=36), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=True),
            sa.Column("surface_id", sa.String(length=20), nullable=True),
            sa.Column("surface_instance_id", sa.String(length=100), nullable=False, server_default=""),
            sa.Column("segment_kind", sa.String(length=20), nullable=False, server_default="manual_break"),
            sa.Column("state", sa.String(length=20), nullable=False, server_default="open"),
            sa.Column("start_message_id", sa.String(length=36), nullable=True),
            sa.Column("end_message_id", sa.String(length=36), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
            sa.Column("usefulness", sa.String(length=20), nullable=False, server_default="unknown"),
            sa.Column("summary_text", sa.Text(), nullable=True),
            sa.Column("key_events", sa.JSON(), nullable=True),
            sa.Column("open_threads", sa.JSON(), nullable=True),
            sa.Column("participants", sa.JSON(), nullable=True),
            sa.Column("summary_model", sa.String(length=120), nullable=True),
            sa.Column("summary_prompt_version", sa.String(length=80), nullable=True),
            sa.Column("summary_input_hash", sa.String(length=64), nullable=True),
            sa.Column("summary_created_at", sa.DateTime(), nullable=True),
            sa.Column("summary_vector_id", sa.String(length=64), nullable=True),
            sa.Column("embedding_model", sa.String(length=100), nullable=True),
            sa.Column("resume_source_segment_id", sa.String(length=36), nullable=True),
            sa.Column("resume_recap_injected_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_conversation_id"):
        op.create_index("ix_conversation_segments_conversation_id", "conversation_segments", ["conversation_id"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_relationship_id"):
        op.create_index("ix_conversation_segments_relationship_id", "conversation_segments", ["relationship_id"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_surface_id"):
        op.create_index("ix_conversation_segments_surface_id", "conversation_segments", ["surface_id"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_segment_kind"):
        op.create_index("ix_conversation_segments_segment_kind", "conversation_segments", ["segment_kind"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_state"):
        op.create_index("ix_conversation_segments_state", "conversation_segments", ["state"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_start_message_id"):
        op.create_index("ix_conversation_segments_start_message_id", "conversation_segments", ["start_message_id"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_end_message_id"):
        op.create_index("ix_conversation_segments_end_message_id", "conversation_segments", ["end_message_id"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_started_at"):
        op.create_index("ix_conversation_segments_started_at", "conversation_segments", ["started_at"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_ended_at"):
        op.create_index("ix_conversation_segments_ended_at", "conversation_segments", ["ended_at"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_usefulness"):
        op.create_index("ix_conversation_segments_usefulness", "conversation_segments", ["usefulness"])
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_summary_prompt_version"):
        op.create_index(
            "ix_conversation_segments_summary_prompt_version",
            "conversation_segments",
            ["summary_prompt_version"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_summary_input_hash"):
        op.create_index(
            "ix_conversation_segments_summary_input_hash",
            "conversation_segments",
            ["summary_input_hash"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_summary_vector_id"):
        op.create_index(
            "ix_conversation_segments_summary_vector_id",
            "conversation_segments",
            ["summary_vector_id"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_conversation_started"):
        op.create_index(
            "ix_conversation_segments_conversation_started",
            "conversation_segments",
            ["conversation_id", "started_at"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_conversation_state_started"):
        op.create_index(
            "ix_conversation_segments_conversation_state_started",
            "conversation_segments",
            ["conversation_id", "state", "started_at"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_conversation_ended"):
        op.create_index(
            "ix_conversation_segments_conversation_ended",
            "conversation_segments",
            ["conversation_id", "ended_at"],
        )
    if not _has_index(inspector, "conversation_segments", "ix_conversation_segments_relationship_surface_started"):
        op.create_index(
            "ix_conversation_segments_relationship_surface_started",
            "conversation_segments",
            ["relationship_id", "surface_id", "surface_instance_id", "started_at"],
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "conversation_segments"):
        index_names = (
            "ix_conversation_segments_relationship_surface_started",
            "ix_conversation_segments_conversation_ended",
            "ix_conversation_segments_conversation_state_started",
            "ix_conversation_segments_conversation_started",
            "ix_conversation_segments_summary_vector_id",
            "ix_conversation_segments_summary_input_hash",
            "ix_conversation_segments_summary_prompt_version",
            "ix_conversation_segments_usefulness",
            "ix_conversation_segments_ended_at",
            "ix_conversation_segments_started_at",
            "ix_conversation_segments_end_message_id",
            "ix_conversation_segments_start_message_id",
            "ix_conversation_segments_state",
            "ix_conversation_segments_segment_kind",
            "ix_conversation_segments_surface_id",
            "ix_conversation_segments_relationship_id",
            "ix_conversation_segments_conversation_id",
        )
        for index_name in index_names:
            inspector = sa.inspect(bind)
            if _has_index(inspector, "conversation_segments", index_name):
                op.drop_index(index_name, table_name="conversation_segments")
        op.drop_table("conversation_segments")
