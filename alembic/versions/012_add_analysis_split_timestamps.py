"""Add separate analysis timestamps for summaries and memories

Revision ID: 012_add_analysis_split_timestamps
Revises: 011_add_memory_durability_and_summary_open_questions
Create Date: 2026-02-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "012_add_analysis_split_timestamps"
down_revision = "011_add_memory_durability_and_summary_open_questions"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "conversations",
        sa.Column("last_summary_analyzed_at", sa.DateTime, nullable=True)
    )
    op.add_column(
        "conversations",
        sa.Column("last_memories_analyzed_at", sa.DateTime, nullable=True)
    )


def downgrade():
    op.drop_column("conversations", "last_memories_analyzed_at")
    op.drop_column("conversations", "last_summary_analyzed_at")
