"""Add source column to memories table

Revision ID: 009_add_memory_source
Revises: 008_add_conversation_source
Create Date: 2026-01-20

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '009_add_memory_source'
down_revision = '008_add_conversation_source'
branch_labels = None
depends_on = None


def upgrade():
    # Add source column to memories table (default 'web' for existing memories)
    op.add_column('memories', sa.Column('source', sa.String(20), nullable=False, server_default='web'))


def downgrade():
    # Remove source column
    op.drop_column('memories', 'source')
