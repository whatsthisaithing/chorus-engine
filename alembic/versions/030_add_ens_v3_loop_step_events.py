"""Add ENS v3 loop step events table.

Revision ID: 030_add_ens_v3_loop_step_events
Revises: 029_add_ens_v3_loop_sessions
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "030_add_ens_v3_loop_step_events"
down_revision = "029_add_ens_v3_loop_sessions"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "ens_loop_step_events"):
        op.create_table(
            "ens_loop_step_events",
            sa.Column("event_id", sa.String(length=36), nullable=False),
            sa.Column("loop_id", sa.String(length=36), nullable=False),
            sa.Column("signal_id", sa.String(length=36), nullable=True),
            sa.Column("tick_id", sa.String(length=36), nullable=True),
            sa.Column("decision_id", sa.String(length=36), nullable=True),
            sa.Column("action_id", sa.String(length=36), nullable=True),
            sa.Column("relationship_id", sa.String(length=36), nullable=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=True),
            sa.Column("surface_id", sa.String(length=20), nullable=True),
            sa.Column("step_index_before", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("step_index_after", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("step_count_after", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("state_before", sa.String(length=30), nullable=True),
            sa.Column("state_after", sa.String(length=30), nullable=True),
            sa.Column("control_action", sa.String(length=30), nullable=True),
            sa.Column("tool_requests_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("provider_finish_reason", sa.String(length=30), nullable=True),
            sa.Column("output_json", sa.JSON(), nullable=True),
            sa.Column("created_at_us", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.PrimaryKeyConstraint("event_id"),
        )

    inspector = sa.inspect(bind)
    for idx_name, cols in [
        ("ix_ens_loop_step_events_loop_id", ["loop_id"]),
        ("ix_ens_loop_step_events_signal_id", ["signal_id"]),
        ("ix_ens_loop_step_events_tick_id", ["tick_id"]),
        ("ix_ens_loop_step_events_decision_id", ["decision_id"]),
        ("ix_ens_loop_step_events_action_id", ["action_id"]),
        ("ix_ens_loop_step_events_relationship_id", ["relationship_id"]),
        ("ix_ens_loop_step_events_conversation_id", ["conversation_id"]),
        ("ix_ens_loop_step_events_surface_id", ["surface_id"]),
        ("ix_ens_loop_step_events_state_before", ["state_before"]),
        ("ix_ens_loop_step_events_state_after", ["state_after"]),
        ("ix_ens_loop_step_events_control_action", ["control_action"]),
        ("ix_ens_loop_step_events_created_at_us", ["created_at_us"]),
        ("ix_ens_loop_step_events_loop_created", ["loop_id", "created_at_us"]),
        ("ix_ens_loop_step_events_signal", ["signal_id", "created_at_us"]),
    ]:
        if not _has_index(inspector, "ens_loop_step_events", idx_name):
            op.create_index(idx_name, "ens_loop_step_events", cols)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events"):
        for idx_name in [
            "ix_ens_loop_step_events_signal",
            "ix_ens_loop_step_events_loop_created",
            "ix_ens_loop_step_events_created_at_us",
            "ix_ens_loop_step_events_control_action",
            "ix_ens_loop_step_events_state_after",
            "ix_ens_loop_step_events_state_before",
            "ix_ens_loop_step_events_surface_id",
            "ix_ens_loop_step_events_conversation_id",
            "ix_ens_loop_step_events_relationship_id",
            "ix_ens_loop_step_events_action_id",
            "ix_ens_loop_step_events_decision_id",
            "ix_ens_loop_step_events_tick_id",
            "ix_ens_loop_step_events_signal_id",
            "ix_ens_loop_step_events_loop_id",
        ]:
            if _has_index(inspector, "ens_loop_step_events", idx_name):
                op.drop_index(idx_name, table_name="ens_loop_step_events")
        op.drop_table("ens_loop_step_events")
