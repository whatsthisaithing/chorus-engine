"""Add surface bindings for ENS Slice 6 routing ownership.

Revision ID: 021_add_surface_bindings
Revises: 020_add_ens_slice3_continuity_fields
Create Date: 2026-02-18
"""

from alembic import op
import sqlalchemy as sa


revision = "021_add_surface_bindings"
down_revision = "020_add_ens_slice3_continuity_fields"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "surface_bindings"):
        op.create_table(
            "surface_bindings",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("surface_id", sa.String(length=20), nullable=False),
            sa.Column("surface_instance_id", sa.String(length=100), nullable=False, server_default=""),
            sa.Column("external_thread_id", sa.String(length=255), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=False),
            sa.Column("thread_id", sa.String(length=36), nullable=False),
            sa.Column("owner_user_id", sa.String(length=200), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_surface_id"):
        op.create_index("ix_surface_bindings_surface_id", "surface_bindings", ["surface_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_surface_instance_id"):
        op.create_index("ix_surface_bindings_surface_instance_id", "surface_bindings", ["surface_instance_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_external_thread_id"):
        op.create_index("ix_surface_bindings_external_thread_id", "surface_bindings", ["external_thread_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_relationship_id"):
        op.create_index("ix_surface_bindings_relationship_id", "surface_bindings", ["relationship_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_conversation_id"):
        op.create_index("ix_surface_bindings_conversation_id", "surface_bindings", ["conversation_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_thread_id"):
        op.create_index("ix_surface_bindings_thread_id", "surface_bindings", ["thread_id"])
    if not _has_index(inspector, "surface_bindings", "ix_surface_bindings_owner_user_id"):
        op.create_index("ix_surface_bindings_owner_user_id", "surface_bindings", ["owner_user_id"])
    if not _has_index(inspector, "surface_bindings", "uq_surface_bindings_lookup"):
        op.create_index(
            "uq_surface_bindings_lookup",
            "surface_bindings",
            ["surface_id", "surface_instance_id", "external_thread_id"],
            unique=True,
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_table(inspector, "surface_bindings"):
        for index_name in (
            "uq_surface_bindings_lookup",
            "ix_surface_bindings_owner_user_id",
            "ix_surface_bindings_thread_id",
            "ix_surface_bindings_conversation_id",
            "ix_surface_bindings_relationship_id",
            "ix_surface_bindings_external_thread_id",
            "ix_surface_bindings_surface_instance_id",
            "ix_surface_bindings_surface_id",
        ):
            if _has_index(inspector, "surface_bindings", index_name):
                op.drop_index(index_name, table_name="surface_bindings")
                inspector = sa.inspect(bind)
        op.drop_table("surface_bindings")
