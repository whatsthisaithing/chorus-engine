"""Add ENS runtime session and observability tables.

Revision ID: 018_add_ens_runtime_tables
Revises: 017_add_character_backup_state
Create Date: 2026-02-14
"""

from alembic import op
import sqlalchemy as sa


revision = "018_add_ens_runtime_tables"
down_revision = "017_add_character_backup_state"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ens_sessions",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("assistant_id", sa.String(length=50), nullable=False),
        sa.Column("user_id", sa.String(length=200), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
        sa.Column("thread_id", sa.String(length=36), nullable=True),
        sa.Column("surface", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("external_session_key", sa.String(length=200), nullable=True),
        sa.Column("latency_sensitive", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_signal_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_ens_sessions_assistant_id", "ens_sessions", ["assistant_id"])
    op.create_index("ix_ens_sessions_user_id", "ens_sessions", ["user_id"])
    op.create_index("ix_ens_sessions_conversation_id", "ens_sessions", ["conversation_id"])
    op.create_index("ix_ens_sessions_thread_id", "ens_sessions", ["thread_id"])
    op.create_index("ix_ens_sessions_external_session_key", "ens_sessions", ["external_session_key"])
    op.create_index(
        "ix_ens_sessions_surface_source_thread",
        "ens_sessions",
        ["surface", "source", "thread_id"],
        unique=True,
    )

    op.create_table(
        "ens_decisions",
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("trace_id", sa.String(length=36), nullable=True),
        sa.Column("signal_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=True),
        sa.Column("assistant_id", sa.String(length=50), nullable=True),
        sa.Column("user_id", sa.String(length=200), nullable=True),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("signal_type", sa.String(length=100), nullable=False),
        sa.Column("appraisal_json", sa.JSON(), nullable=True),
        sa.Column("constraints_json", sa.JSON(), nullable=False),
        sa.Column("intent_proposals_json", sa.JSON(), nullable=False),
        sa.Column("arbitration_json", sa.JSON(), nullable=True),
        sa.Column("actions_json", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("decision_id"),
    )
    op.create_index("ix_ens_decisions_trace_id", "ens_decisions", ["trace_id"])
    op.create_index("ix_ens_decisions_signal_id", "ens_decisions", ["signal_id"])
    op.create_index("ix_ens_decisions_session_id", "ens_decisions", ["session_id"])
    op.create_index("ix_ens_decisions_assistant_id", "ens_decisions", ["assistant_id"])
    op.create_index("ix_ens_decisions_user_id", "ens_decisions", ["user_id"])

    op.create_table(
        "ens_action_results",
        sa.Column("action_result_id", sa.String(length=36), nullable=False),
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("action_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("kind", sa.String(length=100), nullable=False),
        sa.Column("execution_class", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("output_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["decision_id"], ["ens_decisions.decision_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("action_result_id"),
    )
    op.create_index("ix_ens_action_results_decision_id", "ens_action_results", ["decision_id"])
    op.create_index("ix_ens_action_results_action_id", "ens_action_results", ["action_id"])
    op.create_index("ix_ens_action_results_idempotency_key", "ens_action_results", ["idempotency_key"])


def downgrade():
    op.drop_index("ix_ens_action_results_idempotency_key", table_name="ens_action_results")
    op.drop_index("ix_ens_action_results_action_id", table_name="ens_action_results")
    op.drop_index("ix_ens_action_results_decision_id", table_name="ens_action_results")
    op.drop_table("ens_action_results")

    op.drop_index("ix_ens_decisions_user_id", table_name="ens_decisions")
    op.drop_index("ix_ens_decisions_assistant_id", table_name="ens_decisions")
    op.drop_index("ix_ens_decisions_session_id", table_name="ens_decisions")
    op.drop_index("ix_ens_decisions_signal_id", table_name="ens_decisions")
    op.drop_index("ix_ens_decisions_trace_id", table_name="ens_decisions")
    op.drop_table("ens_decisions")

    op.drop_index("ix_ens_sessions_surface_source_thread", table_name="ens_sessions")
    op.drop_index("ix_ens_sessions_external_session_key", table_name="ens_sessions")
    op.drop_index("ix_ens_sessions_thread_id", table_name="ens_sessions")
    op.drop_index("ix_ens_sessions_conversation_id", table_name="ens_sessions")
    op.drop_index("ix_ens_sessions_user_id", table_name="ens_sessions")
    op.drop_index("ix_ens_sessions_assistant_id", table_name="ens_sessions")
    op.drop_table("ens_sessions")
