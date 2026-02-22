"""Add ENS v3 floor control state table.

Revision ID: 028_add_ens_v3_floor_control_state
Revises: 027_add_ens_v3_scheduler_hardening
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "028_add_ens_v3_floor_control_state"
down_revision = "027_add_ens_v3_scheduler_hardening"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "ens_floor_control_state"):
        op.create_table(
            "ens_floor_control_state",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("relationship_id", sa.String(length=36), nullable=False),
            sa.Column("active_surface_id", sa.String(length=20), nullable=True),
            sa.Column("attention_lock_until_us", sa.Integer(), nullable=True),
            sa.Column("lock_source_signal_id", sa.String(length=36), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_floor_control_state", "uq_ens_floor_control_state_relationship_id"):
        op.create_index(
            "uq_ens_floor_control_state_relationship_id",
            "ens_floor_control_state",
            ["relationship_id"],
            unique=True,
        )
    if not _has_index(inspector, "ens_floor_control_state", "ix_ens_floor_control_state_surface_lock"):
        op.create_index(
            "ix_ens_floor_control_state_surface_lock",
            "ens_floor_control_state",
            ["active_surface_id", "attention_lock_until_us"],
        )
    if not _has_index(inspector, "ens_floor_control_state", "ix_ens_floor_control_state_relationship_id"):
        op.create_index(
            "ix_ens_floor_control_state_relationship_id",
            "ens_floor_control_state",
            ["relationship_id"],
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ens_floor_control_state"):
        for idx in [
            "ix_ens_floor_control_state_relationship_id",
            "ix_ens_floor_control_state_surface_lock",
            "uq_ens_floor_control_state_relationship_id",
        ]:
            if _has_index(inspector, "ens_floor_control_state", idx):
                op.drop_index(idx, table_name="ens_floor_control_state")
        op.drop_table("ens_floor_control_state")
