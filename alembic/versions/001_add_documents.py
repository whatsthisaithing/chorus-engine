"""Add document analysis tables

Revision ID: 001_add_documents
Revises: 
Create Date: 2025-01-28 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '001_add_documents'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add document analysis tables."""
    
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=False),
        sa.Column('storage_key', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('author', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('last_accessed', sa.DateTime(), nullable=True),
        sa.Column('processing_status', sa.String(length=50), nullable=False),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('vector_collection_id', sa.String(length=200), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('storage_key')
    )
    
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_id', sa.String(length=200), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_length', sa.Integer(), nullable=False),
        sa.Column('page_numbers', sa.String(length=200), nullable=True),
        sa.Column('start_line', sa.Integer(), nullable=True),
        sa.Column('end_line', sa.Integer(), nullable=True),
        sa.Column('chunk_method', sa.String(length=50), nullable=False),
        sa.Column('overlap_tokens', sa.Integer(), nullable=False),
        sa.Column('embedding_model', sa.String(length=100), nullable=True),
        sa.Column('embedding_created_at', sa.DateTime(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chunk_id')
    )
    
    # Create document_access_logs table
    op.create_table(
        'document_access_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.String(length=200), nullable=True),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('access_type', sa.String(length=50), nullable=False),
        sa.Column('chunks_retrieved', sa.Integer(), nullable=False),
        sa.Column('chunk_ids', sa.Text(), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('accessed_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create code_execution_logs table
    op.create_table(
        'code_execution_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.String(length=200), nullable=True),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('language', sa.String(length=50), nullable=False),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('code_hash', sa.String(length=64), nullable=False),
        sa.Column('execution_status', sa.String(length=50), nullable=False),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('stdout', sa.Text(), nullable=True),
        sa.Column('stderr', sa.Text(), nullable=True),
        sa.Column('return_value', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('approved_by_user', sa.Boolean(), nullable=False),
        sa.Column('sandboxed', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('executed_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better query performance
    op.create_index('idx_documents_storage_key', 'documents', ['storage_key'])
    op.create_index('idx_documents_file_type', 'documents', ['file_type'])
    op.create_index('idx_documents_status', 'documents', ['processing_status'])
    op.create_index('idx_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('idx_chunks_chunk_id', 'document_chunks', ['chunk_id'])
    op.create_index('idx_access_logs_document_id', 'document_access_logs', ['document_id'])
    op.create_index('idx_access_logs_conversation_id', 'document_access_logs', ['conversation_id'])
    op.create_index('idx_code_logs_conversation_id', 'code_execution_logs', ['conversation_id'])
    op.create_index('idx_code_logs_status', 'code_execution_logs', ['execution_status'])


def downgrade() -> None:
    """Remove document analysis tables."""
    
    # Drop indexes first
    op.drop_index('idx_code_logs_status', table_name='code_execution_logs')
    op.drop_index('idx_code_logs_conversation_id', table_name='code_execution_logs')
    op.drop_index('idx_access_logs_conversation_id', table_name='document_access_logs')
    op.drop_index('idx_access_logs_document_id', table_name='document_access_logs')
    op.drop_index('idx_chunks_chunk_id', table_name='document_chunks')
    op.drop_index('idx_chunks_document_id', table_name='document_chunks')
    op.drop_index('idx_documents_status', table_name='documents')
    op.drop_index('idx_documents_file_type', table_name='documents')
    op.drop_index('idx_documents_storage_key', table_name='documents')
    
    # Drop tables (in reverse order due to foreign keys)
    op.drop_table('code_execution_logs')
    op.drop_table('document_access_logs')
    op.drop_table('document_chunks')
    op.drop_table('documents')
