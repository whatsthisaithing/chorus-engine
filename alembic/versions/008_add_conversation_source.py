"""Add source field to conversations

Revision ID: 008_add_conversation_source
Revises: 007_fix_downloaded_models
Create Date: 2026-01-20

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008_add_conversation_source'
down_revision = '007_fix_downloaded_models'
branch_labels = None
depends_on = None


def upgrade():
    """Add source column to conversations table."""
    # Add source column (defaults to "web" for existing conversations)
    op.add_column('conversations', sa.Column('source', sa.String(20), nullable=False, server_default='web'))


def downgrade():
    """Remove source column from conversations table."""
    op.drop_column('conversations', 'source')
