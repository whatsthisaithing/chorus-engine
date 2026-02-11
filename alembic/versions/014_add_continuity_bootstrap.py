"""Add continuity bootstrapping tables and conversation fields

Revision ID: 014_add_continuity_bootstrap
Revises: 013_add_message_soft_delete
Create Date: 2026-02-10

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "014_add_continuity_bootstrap"
down_revision = "013_add_message_soft_delete"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "conversations",
        sa.Column("continuity_mode", sa.String(length=10), nullable=False, server_default="ask")
    )
    op.add_column(
        "conversations",
        sa.Column("continuity_choice_remembered", sa.String(length=10), nullable=False, server_default="false")
    )
    op.add_column(
        "conversations",
        sa.Column("primary_user", sa.String(length=200), nullable=True)
    )

    op.create_table(
        "continuity_relationship_states",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("character_id", sa.String(length=50), nullable=False, index=True),
        sa.Column("familiarity_level", sa.String(length=20), nullable=False, server_default="new"),
        sa.Column("tone_baseline", sa.JSON(), nullable=False),
        sa.Column("interaction_contract", sa.JSON(), nullable=False),
        sa.Column("boundaries", sa.JSON(), nullable=False),
        sa.Column("assistant_role_frame", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "continuity_arcs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("character_id", sa.String(length=50), nullable=False, index=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False, server_default="theme"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("confidence", sa.String(length=10), nullable=False, server_default="medium"),
        sa.Column("stickiness", sa.String(length=10), nullable=False, server_default="normal"),
        sa.Column("last_touched_conversation_id", sa.String(length=36), nullable=True),
        sa.Column("last_touched_conversation_at", sa.DateTime(), nullable=True),
        sa.Column("frequency_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "continuity_bootstrap_cache",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("character_id", sa.String(length=50), nullable=False, index=True),
        sa.Column("bootstrap_packet_internal", sa.Text(), nullable=False),
        sa.Column("bootstrap_packet_user_preview", sa.Text(), nullable=False),
        sa.Column("bootstrap_generated_at", sa.DateTime(), nullable=True),
        sa.Column("bootstrap_inputs_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

def downgrade():
    op.drop_table("continuity_bootstrap_cache")
    op.drop_table("continuity_arcs")
    op.drop_table("continuity_relationship_states")
    op.drop_column("conversations", "primary_user")
    op.drop_column("conversations", "continuity_choice_remembered")
    op.drop_column("conversations", "continuity_mode")
