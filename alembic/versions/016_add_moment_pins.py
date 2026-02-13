"""Add moment pins table

Revision ID: 016_add_moment_pins
Revises: 015_add_media_offer_state
Create Date: 2026-02-13
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "016_add_moment_pins"
down_revision = "015_add_media_offer_state"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "moment_pins",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=200), nullable=False),
        sa.Column("character_id", sa.String(length=50), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("selected_message_ids", sa.JSON(), nullable=False),
        sa.Column("transcript_snapshot", sa.Text(), nullable=False),
        sa.Column("what_happened", sa.Text(), nullable=False),
        sa.Column("why_model", sa.Text(), nullable=False),
        sa.Column("why_user", sa.Text(), nullable=True),
        sa.Column("quote_snippet", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("reinforcement_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("turns_since_reinforcement", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("archived", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("telemetry_flags", sa.JSON(), nullable=False),
        sa.Column("vector_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_moment_pins_user_id", "moment_pins", ["user_id"], unique=False)
    op.create_index("ix_moment_pins_character_id", "moment_pins", ["character_id"], unique=False)
    op.create_index("ix_moment_pins_conversation_id", "moment_pins", ["conversation_id"], unique=False)
    op.create_index("ix_moment_pins_created_at", "moment_pins", ["created_at"], unique=False)
    op.create_index("ix_moment_pins_archived", "moment_pins", ["archived"], unique=False)


def downgrade():
    op.drop_index("ix_moment_pins_archived", table_name="moment_pins")
    op.drop_index("ix_moment_pins_created_at", table_name="moment_pins")
    op.drop_index("ix_moment_pins_conversation_id", table_name="moment_pins")
    op.drop_index("ix_moment_pins_character_id", table_name="moment_pins")
    op.drop_index("ix_moment_pins_user_id", table_name="moment_pins")
    op.drop_table("moment_pins")
