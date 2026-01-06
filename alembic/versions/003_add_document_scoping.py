"""Add document scoping for conversation-level isolation

Revision ID: 003_add_document_scoping
Revises: 002_add_character_scoping
Create Date: 2026-01-05

Adds scoping fields to documents table to enable conversation-level
document isolation for privacy and multi-project work.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_add_document_scoping'
down_revision = '002_add_character_scoping'
branch_labels = None
depends_on = None


def upgrade():
    """Add document scoping fields."""
    
    # Add scoping columns to documents table
    op.add_column('documents', 
        sa.Column('document_scope', sa.String(50), nullable=False, server_default='conversation')
    )
    op.add_column('documents',
        sa.Column('conversation_id', sa.String(200), nullable=True)
    )
    
    # Create index for efficient scope queries
    op.create_index(
        'idx_documents_scope',
        'documents',
        ['document_scope', 'conversation_id', 'character_id']
    )
    
    # Update existing documents to character scope (migration safety)
    # This is safer than conversation scope for existing documents
    # Users can re-upload if they need conversation isolation
    op.execute("""
        UPDATE documents 
        SET document_scope = 'character' 
        WHERE document_scope = 'conversation' AND conversation_id IS NULL
    """)


def downgrade():
    """Remove document scoping fields."""
    
    # Drop index
    op.drop_index('idx_documents_scope', table_name='documents')
    
    # Drop columns
    op.drop_column('documents', 'conversation_id')
    op.drop_column('documents', 'document_scope')
