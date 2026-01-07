"""Add video confirmation preference

Revision ID: 005_add_video_confirmation
Revises: 004_add_videos
Create Date: 2026-01-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005_add_video_confirmation'
down_revision = '004_add_videos'
branch_labels = None
depends_on = None


def upgrade():
    """Add video_confirmation_disabled column to conversations table."""
    # Add video_confirmation_disabled column (defaults to "false" - confirmations enabled)
    op.add_column('conversations', sa.Column('video_confirmation_disabled', sa.String(10), nullable=False, server_default='false'))


def downgrade():
    """Remove video_confirmation_disabled column from conversations table."""
    op.drop_column('conversations', 'video_confirmation_disabled')
