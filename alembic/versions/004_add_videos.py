"""Add videos table for video generation support

Revision ID: 004_add_videos
Revises: 003_add_document_scoping
Create Date: 2026-01-06 10:30:00

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '004_add_videos'
down_revision = '003_add_document_scoping'
branch_labels = None
depends_on = None


def upgrade():
    """Add videos table for video generation tracking."""
    op.create_table(
        'videos',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('conversation_id', sa.String(36), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('thumbnail_path', sa.String(500), nullable=True),
        sa.Column('format', sa.String(10), nullable=True),  # webm, mp4, webp, etc.
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('negative_prompt', sa.Text(), nullable=True),
        sa.Column('workflow_file', sa.String(500), nullable=True),
        sa.Column('comfy_prompt_id', sa.String(100), nullable=True),
        sa.Column('generation_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        
        # Foreign key to conversations
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE')
    )
    
    # Create index for faster conversation video lookups
    op.create_index('idx_videos_conversation', 'videos', ['conversation_id'])


def downgrade():
    """Remove videos table."""
    op.drop_index('idx_videos_conversation', table_name='videos')
    op.drop_table('videos')
