"""Relationship-first v0 schema changes.

Revision ID: 023_relationship_first_v0
Revises: 022_add_surface_egress_intents
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa


revision = "023_relationship_first_v0"
down_revision = "022_add_surface_egress_intents"
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

    if not _has_table(inspector, "relationships"):
        op.create_table(
            "relationships",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("owner_user_id", sa.String(length=200), nullable=False),
            sa.Column("character_id", sa.String(length=50), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("last_interaction_at", sa.DateTime(), nullable=True),
            sa.Column("state_version", sa.Integer(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "relationships", "ix_relationships_owner_user_id"):
        op.create_index("ix_relationships_owner_user_id", "relationships", ["owner_user_id"])
    if not _has_index(inspector, "relationships", "ix_relationships_character_id"):
        op.create_index("ix_relationships_character_id", "relationships", ["character_id"])
    if not _has_index(inspector, "relationships", "uq_relationships_owner_character"):
        op.create_index(
            "uq_relationships_owner_character",
            "relationships",
            ["owner_user_id", "character_id"],
            unique=True,
        )

    if not _has_table(inspector, "relationship_surfaces"):
        op.create_table(
            "relationship_surfaces",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=False),
            sa.Column("surface_id", sa.String(length=20), nullable=False),
            sa.Column("surface_instance_id", sa.String(length=100), nullable=False, server_default=""),
            sa.Column("general_conversation_id", sa.String(length=36), nullable=True),
            sa.Column("last_bootstrap_seen_fingerprint", sa.String(length=128), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("last_interaction_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["relationship_id"], ["relationships.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "relationship_surfaces", "ix_relationship_surfaces_relationship_id"):
        op.create_index("ix_relationship_surfaces_relationship_id", "relationship_surfaces", ["relationship_id"])
    if not _has_index(inspector, "relationship_surfaces", "ix_relationship_surfaces_surface_id"):
        op.create_index("ix_relationship_surfaces_surface_id", "relationship_surfaces", ["surface_id"])
    if not _has_index(inspector, "relationship_surfaces", "ix_relationship_surfaces_surface_instance_id"):
        op.create_index(
            "ix_relationship_surfaces_surface_instance_id",
            "relationship_surfaces",
            ["surface_instance_id"],
        )
    if not _has_index(inspector, "relationship_surfaces", "ix_relationship_surfaces_general_conversation_id"):
        op.create_index(
            "ix_relationship_surfaces_general_conversation_id",
            "relationship_surfaces",
            ["general_conversation_id"],
        )
    if not _has_index(inspector, "relationship_surfaces", "uq_relationship_surfaces_lookup"):
        op.create_index(
            "uq_relationship_surfaces_lookup",
            "relationship_surfaces",
            ["relationship_id", "surface_id", "surface_instance_id"],
            unique=True,
        )

    inspector = sa.inspect(bind)
    if not _has_column(inspector, "conversations", "relationship_id"):
        op.add_column("conversations", sa.Column("relationship_id", sa.String(length=36), nullable=True))
    if not _has_column(inspector, "conversations", "conversation_kind"):
        op.add_column("conversations", sa.Column("conversation_kind", sa.String(length=30), nullable=False, server_default="standard"))
    if not _has_column(inspector, "conversations", "general_chat_memories_processed_through_message_id"):
        op.add_column(
            "conversations",
            sa.Column("general_chat_memories_processed_through_message_id", sa.String(length=36), nullable=True),
        )
    if not _has_column(inspector, "conversations", "general_chat_memories_processed_through_created_at"):
        op.add_column(
            "conversations",
            sa.Column("general_chat_memories_processed_through_created_at", sa.DateTime(), nullable=True),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "conversations", "ix_conversations_relationship_id"):
        op.create_index("ix_conversations_relationship_id", "conversations", ["relationship_id"])
    if not _has_index(inspector, "conversations", "ix_conversations_conversation_kind"):
        op.create_index("ix_conversations_conversation_kind", "conversations", ["conversation_kind"])
    if not _has_index(inspector, "conversations", "ix_conversations_relationship_kind"):
        op.create_index(
            "ix_conversations_relationship_kind",
            "conversations",
            ["relationship_id", "conversation_kind"],
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_index(inspector, "conversations", "ix_conversations_relationship_kind"):
        op.drop_index("ix_conversations_relationship_kind", table_name="conversations")
    if _has_index(inspector, "conversations", "ix_conversations_conversation_kind"):
        op.drop_index("ix_conversations_conversation_kind", table_name="conversations")
    if _has_index(inspector, "conversations", "ix_conversations_relationship_id"):
        op.drop_index("ix_conversations_relationship_id", table_name="conversations")
    if _has_column(inspector, "conversations", "general_chat_memories_processed_through_created_at"):
        op.drop_column("conversations", "general_chat_memories_processed_through_created_at")
    if _has_column(inspector, "conversations", "general_chat_memories_processed_through_message_id"):
        op.drop_column("conversations", "general_chat_memories_processed_through_message_id")
    if _has_column(inspector, "conversations", "conversation_kind"):
        op.drop_column("conversations", "conversation_kind")
    if _has_column(inspector, "conversations", "relationship_id"):
        op.drop_column("conversations", "relationship_id")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "relationship_surfaces"):
        for index_name in (
            "uq_relationship_surfaces_lookup",
            "ix_relationship_surfaces_general_conversation_id",
            "ix_relationship_surfaces_surface_instance_id",
            "ix_relationship_surfaces_surface_id",
            "ix_relationship_surfaces_relationship_id",
        ):
            if _has_index(inspector, "relationship_surfaces", index_name):
                op.drop_index(index_name, table_name="relationship_surfaces")
                inspector = sa.inspect(bind)
        op.drop_table("relationship_surfaces")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "relationships"):
        for index_name in (
            "uq_relationships_owner_character",
            "ix_relationships_character_id",
            "ix_relationships_owner_user_id",
        ):
            if _has_index(inspector, "relationships", index_name):
                op.drop_index(index_name, table_name="relationships")
                inspector = sa.inspect(bind)
        op.drop_table("relationships")
