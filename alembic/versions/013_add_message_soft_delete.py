"""Add soft delete timestamp to messages

Revision ID: 013_add_message_soft_delete
Revises: 012_add_analysis_split_timestamps
Create Date: 2026-02-09

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "013_add_message_soft_delete"
down_revision = "012_add_analysis_split_timestamps"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "messages",
        sa.Column("deleted_at", sa.DateTime(), nullable=True)
    )


def downgrade():
    op.drop_column("messages", "deleted_at")
