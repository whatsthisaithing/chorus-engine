"""Add memory durability fields and summary open_questions

Revision ID: 011_add_memory_durability_and_summary_open_questions
Revises: 010_add_image_attachments
Create Date: 2026-02-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "011_add_memory_durability_and_summary_open_questions"
down_revision = "010_add_image_attachments"
branch_labels = None
depends_on = None


def upgrade():
    # Add durability and pattern eligibility to memories
    op.add_column(
        "memories",
        sa.Column("durability", sa.String(20), nullable=False, server_default="situational")
    )
    op.add_column(
        "memories",
        sa.Column("pattern_eligible", sa.Integer, nullable=False, server_default="0")
    )
    
    # Add open_questions to conversation summaries
    op.add_column(
        "conversation_summaries",
        sa.Column("open_questions", sa.JSON, nullable=True)
    )


def downgrade():
    op.drop_column("conversation_summaries", "open_questions")
    op.drop_column("memories", "pattern_eligible")
    op.drop_column("memories", "durability")
