"""Add ENS tool call request table.

Revision ID: 019_add_ens_tool_call_requests
Revises: 018_add_ens_runtime_tables
Create Date: 2026-02-14
"""

from alembic import op
import sqlalchemy as sa


revision = "019_add_ens_tool_call_requests"
down_revision = "018_add_ens_runtime_tables"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ens_tool_call_requests",
        sa.Column("tool_call_id", sa.String(length=100), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("assistant_message_id", sa.String(length=36), nullable=True),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("args_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("result_ref", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("tool_call_id"),
    )
    op.create_index("ix_ens_tool_call_requests_session_id", "ens_tool_call_requests", ["session_id"])
    op.create_index("ix_ens_tool_call_requests_assistant_message_id", "ens_tool_call_requests", ["assistant_message_id"])
    op.create_index("ix_ens_tool_call_requests_tool_name", "ens_tool_call_requests", ["tool_name"])
    op.create_index("ix_ens_tool_call_requests_status", "ens_tool_call_requests", ["status"])
    op.create_index("ix_ens_tool_call_requests_idempotency_key", "ens_tool_call_requests", ["idempotency_key"])
    op.create_index(
        "ix_ens_tool_call_requests_session_status",
        "ens_tool_call_requests",
        ["session_id", "status"],
    )


def downgrade():
    op.drop_index("ix_ens_tool_call_requests_session_status", table_name="ens_tool_call_requests")
    op.drop_index("ix_ens_tool_call_requests_idempotency_key", table_name="ens_tool_call_requests")
    op.drop_index("ix_ens_tool_call_requests_status", table_name="ens_tool_call_requests")
    op.drop_index("ix_ens_tool_call_requests_tool_name", table_name="ens_tool_call_requests")
    op.drop_index("ix_ens_tool_call_requests_assistant_message_id", table_name="ens_tool_call_requests")
    op.drop_index("ix_ens_tool_call_requests_session_id", table_name="ens_tool_call_requests")
    op.drop_table("ens_tool_call_requests")
