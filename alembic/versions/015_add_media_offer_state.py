"""Add conversation-level media offer state

Revision ID: 015_add_media_offer_state
Revises: 014_add_continuity_bootstrap
Create Date: 2026-02-11
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "015_add_media_offer_state"
down_revision = "014_add_continuity_bootstrap"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "conversations",
        sa.Column("allow_image_offers", sa.String(10), nullable=False, server_default="true"),
    )
    op.add_column(
        "conversations",
        sa.Column("allow_video_offers", sa.String(10), nullable=False, server_default="true"),
    )
    op.add_column(
        "conversations",
        sa.Column("last_image_offer_at", sa.DateTime, nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("last_video_offer_at", sa.DateTime, nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("last_image_offer_message_count", sa.Integer, nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("last_video_offer_message_count", sa.Integer, nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("image_offer_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "conversations",
        sa.Column("video_offer_count", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade():
    op.drop_column("conversations", "video_offer_count")
    op.drop_column("conversations", "image_offer_count")
    op.drop_column("conversations", "last_video_offer_message_count")
    op.drop_column("conversations", "last_image_offer_message_count")
    op.drop_column("conversations", "last_video_offer_at")
    op.drop_column("conversations", "last_image_offer_at")
    op.drop_column("conversations", "allow_video_offers")
    op.drop_column("conversations", "allow_image_offers")
