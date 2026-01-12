"""Add downloaded_models table for unified model tracking

Revision ID: 006_add_custom_models
Revises: 005_add_video_confirmation
Create Date: 2026-01-11

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '006_add_custom_models'
down_revision = '005_add_video_confirmation'
branch_labels = None
depends_on = None


def upgrade():
    """Add downloaded_models table for unified tracking of all models."""
    op.create_table(
        'downloaded_models',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.String(length=200), nullable=False),
        sa.Column('display_name', sa.String(length=500), nullable=False),
        sa.Column('repo_id', sa.String(length=500), nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=True),
        sa.Column('quantization', sa.String(length=100), nullable=False),
        sa.Column('parameters', sa.Float(), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=True),
        sa.Column('file_size_mb', sa.Float(), nullable=True),
        sa.Column('file_path', sa.String(length=1000), nullable=True),
        sa.Column('ollama_model_name', sa.String(length=500), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=False),  # 'curated' or 'custom_hf'
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('downloaded_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    
    # Create indexes for faster lookups
    op.create_index('idx_downloaded_models_model_id', 'downloaded_models', ['model_id'])
    op.create_index('idx_downloaded_models_source', 'downloaded_models', ['source'])
    op.create_index('idx_downloaded_models_ollama_name', 'downloaded_models', ['ollama_model_name'])


def downgrade():
    """Remove downloaded_models table."""
    op.drop_index('idx_downloaded_models_ollama_name', table_name='downloaded_models')
    op.drop_index('idx_downloaded_models_source', table_name='downloaded_models')
    op.drop_index('idx_downloaded_models_model_id', table_name='downloaded_models')
    op.drop_table('downloaded_models')
