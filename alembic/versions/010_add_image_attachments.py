"""Add image_attachments table for vision system

Revision ID: 010_add_image_attachments
Revises: 009_add_memory_source
Create Date: 2026-01-25

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '010_add_image_attachments'
down_revision = '009_add_memory_source'
branch_labels = None
depends_on = None


def upgrade():
    # Create image_attachments table
    op.create_table(
        'image_attachments',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('message_id', sa.String(50), nullable=False),
        sa.Column('conversation_id', sa.String(50), nullable=False),
        sa.Column('character_id', sa.String(50), nullable=False),
        
        # File storage
        sa.Column('original_path', sa.String(500), nullable=False),
        sa.Column('processed_path', sa.String(500), nullable=True),
        sa.Column('original_filename', sa.String(255), nullable=True),
        sa.Column('file_size', sa.Integer, nullable=True),
        sa.Column('mime_type', sa.String(50), nullable=True),
        sa.Column('width', sa.Integer, nullable=True),
        sa.Column('height', sa.Integer, nullable=True),
        sa.Column('uploaded_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        
        # Vision analysis
        sa.Column('vision_processed', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('vision_skipped', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('vision_skip_reason', sa.String(100), nullable=True),
        
        sa.Column('vision_model', sa.String(100), nullable=True),
        sa.Column('vision_backend', sa.String(50), nullable=True),
        sa.Column('vision_processed_at', sa.DateTime, nullable=True),
        sa.Column('vision_processing_time_ms', sa.Integer, nullable=True),
        
        sa.Column('vision_observation', sa.Text, nullable=True),
        sa.Column('vision_confidence', sa.Float, nullable=True),
        sa.Column('vision_tags', sa.Text, nullable=True),
        
        # User metadata
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('source', sa.String(20), nullable=False, server_default='web'),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['character_id'], ['characters.id'], ondelete='CASCADE')
    )
    
    # Create indexes for common query patterns
    op.create_index('idx_image_attachments_message', 'image_attachments', ['message_id'])
    op.create_index('idx_image_attachments_conversation', 'image_attachments', ['conversation_id'])
    op.create_index('idx_image_attachments_character', 'image_attachments', ['character_id'])
    op.create_index('idx_image_attachments_uploaded', 'image_attachments', ['uploaded_at'])


def downgrade():
    # Drop indexes
    op.drop_index('idx_image_attachments_uploaded', 'image_attachments')
    op.drop_index('idx_image_attachments_character', 'image_attachments')
    op.drop_index('idx_image_attachments_conversation', 'image_attachments')
    op.drop_index('idx_image_attachments_message', 'image_attachments')
    
    # Drop table
    op.drop_table('image_attachments')
