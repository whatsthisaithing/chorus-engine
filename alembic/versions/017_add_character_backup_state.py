"""Add character backup scheduler state table

Revision ID: 017_add_character_backup_state
Revises: 016_add_moment_pins
Create Date: 2026-02-14
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "017_add_character_backup_state"
down_revision = "016_add_moment_pins"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "character_backup_state",
        sa.Column("character_id", sa.String(length=50), nullable=False),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(), nullable=True),
        sa.Column("last_status", sa.String(length=20), nullable=False, server_default="never"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_manifest_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("character_id"),
    )


def downgrade():
    op.drop_table("character_backup_state")
