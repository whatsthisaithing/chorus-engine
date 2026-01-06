"""Add character_id to documents for scoping.

Revision ID: 002_add_character_scoping
Revises: 001_add_documents
Create Date: 2026-01-05
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_character_scoping'
down_revision = '001_add_documents'
branch_labels = None
depends_on = None


def upgrade():
    """Add character_id column to documents table."""
    
    # Add character_id column (nullable for global documents)
    op.add_column(
        'documents',
        sa.Column('character_id', sa.String(100), nullable=True)
    )
    
    # Add index for character_id filtering
    op.create_index(
        'idx_documents_character_id',
        'documents',
        ['character_id']
    )
    
    # Add composite index for character + type filtering
    op.create_index(
        'idx_documents_character_type',
        'documents',
        ['character_id', 'file_type']
    )


def downgrade():
    """Remove character_id column from documents table."""
    
    # Drop indexes
    op.drop_index('idx_documents_character_type', table_name='documents')
    op.drop_index('idx_documents_character_id', table_name='documents')
    
    # Drop column
    op.drop_column('documents', 'character_id')
