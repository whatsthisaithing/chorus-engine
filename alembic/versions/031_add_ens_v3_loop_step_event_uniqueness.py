"""Add uniqueness constraint for loop step events by (loop_id, signal_id).

Revision ID: 031_add_ens_v3_loop_step_event_uniqueness
Revises: 030_add_ens_v3_loop_step_events
Create Date: 2026-02-22
"""

from alembic import op
import sqlalchemy as sa


revision = "031_add_ens_v3_loop_step_event_uniqueness"
down_revision = "030_add_ens_v3_loop_step_events"
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
        return

    # Deduplicate any pre-existing rows so unique index creation is safe.
    duplicate_keys = bind.execute(
        sa.text(
            """
            SELECT loop_id, signal_id
            FROM ens_loop_step_events
            WHERE signal_id IS NOT NULL
            GROUP BY loop_id, signal_id
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()

    for loop_id, signal_id in duplicate_keys:
        rows = bind.execute(
            sa.text(
                """
                SELECT event_id
                FROM ens_loop_step_events
                WHERE loop_id = :loop_id AND signal_id = :signal_id
                ORDER BY created_at_us ASC, event_id ASC
                """
            ),
            {"loop_id": loop_id, "signal_id": signal_id},
        ).fetchall()
        for extra in rows[1:]:
            bind.execute(
                sa.text("DELETE FROM ens_loop_step_events WHERE event_id = :event_id"),
                {"event_id": extra[0]},
            )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "ens_loop_step_events", "uq_ens_loop_step_events_loop_signal"):
        op.create_index(
            "uq_ens_loop_step_events_loop_signal",
            "ens_loop_step_events",
            ["loop_id", "signal_id"],
            unique=True,
            sqlite_where=sa.text("signal_id IS NOT NULL"),
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events") and _has_index(
        inspector,
        "ens_loop_step_events",
        "uq_ens_loop_step_events_loop_signal",
    ):
        op.drop_index("uq_ens_loop_step_events_loop_signal", table_name="ens_loop_step_events")

