"""Add ENS v3 loop memory compression persistence.

Revision ID: 033_add_ens_v3_loop_memory_compression
Revises: 032_add_ens_v3_loop_mode
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa


revision = "033_add_ens_v3_loop_memory_compression"
down_revision = "032_add_ens_v3_loop_mode"
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ens_loop_sessions") and not _has_column(
        inspector, "ens_loop_sessions", "last_compressed_step_index"
    ):
        op.add_column(
            "ens_loop_sessions",
            sa.Column("last_compressed_step_index", sa.Integer(), nullable=False, server_default="-1"),
        )
        op.execute("UPDATE ens_loop_sessions SET last_compressed_step_index = -1 WHERE last_compressed_step_index IS NULL")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_sessions") and not _has_index(
        inspector, "ens_loop_sessions", "ix_ens_loop_sessions_last_compressed_step_index"
    ):
        op.create_index(
            "ix_ens_loop_sessions_last_compressed_step_index",
            "ens_loop_sessions",
            ["last_compressed_step_index"],
        )

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events") and not _has_column(
        inspector, "ens_loop_step_events", "memory_payload_json"
    ):
        op.add_column("ens_loop_step_events", sa.Column("memory_payload_json", sa.JSON(), nullable=True))

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events") and not _has_column(
        inspector, "ens_loop_step_events", "compression_artifact_id"
    ):
        op.add_column("ens_loop_step_events", sa.Column("compression_artifact_id", sa.String(length=36), nullable=True))

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events") and not _has_index(
        inspector, "ens_loop_step_events", "ix_ens_loop_step_events_compression_artifact_id"
    ):
        op.create_index(
            "ix_ens_loop_step_events_compression_artifact_id",
            "ens_loop_step_events",
            ["compression_artifact_id"],
        )

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events") and not _has_index(
        inspector, "ens_loop_step_events", "ix_ens_loop_step_events_loop_step_after"
    ):
        op.create_index(
            "ix_ens_loop_step_events_loop_step_after",
            "ens_loop_step_events",
            ["loop_id", "step_index_after"],
        )

    inspector = sa.inspect(bind)
    if not _has_table(inspector, "ens_loop_compression_artifacts"):
        op.create_table(
            "ens_loop_compression_artifacts",
            sa.Column("artifact_id", sa.String(length=36), nullable=False),
            sa.Column("loop_id", sa.String(length=36), nullable=False),
            sa.Column("from_step_index", sa.Integer(), nullable=False),
            sa.Column("to_step_index", sa.Integer(), nullable=False),
            sa.Column("input_hash", sa.String(length=64), nullable=False),
            sa.Column("config_hash", sa.String(length=64), nullable=False),
            sa.Column("output_hash", sa.String(length=64), nullable=False),
            sa.Column("folded_json", sa.JSON(), nullable=False),
            sa.Column("created_at_us", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(datetime('now'))")),
            sa.PrimaryKeyConstraint("artifact_id"),
        )

    inspector = sa.inspect(bind)
    for idx_name, cols in [
        ("ix_ens_loop_compression_artifacts_loop_id", ["loop_id"]),
        ("ix_ens_loop_compression_artifacts_input_hash", ["input_hash"]),
        ("ix_ens_loop_compression_artifacts_config_hash", ["config_hash"]),
        ("ix_ens_loop_compression_artifacts_output_hash", ["output_hash"]),
        ("ix_ens_loop_compression_artifacts_created_at_us", ["created_at_us"]),
        ("ix_ens_loop_compression_artifacts_loop_to_step", ["loop_id", "to_step_index"]),
        ("ix_ens_loop_compression_artifacts_loop_created", ["loop_id", "created_at_us"]),
    ]:
        if not _has_index(inspector, "ens_loop_compression_artifacts", idx_name):
            op.create_index(idx_name, "ens_loop_compression_artifacts", cols)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "ens_loop_compression_artifacts"):
        for idx_name in [
            "ix_ens_loop_compression_artifacts_loop_created",
            "ix_ens_loop_compression_artifacts_loop_to_step",
            "ix_ens_loop_compression_artifacts_created_at_us",
            "ix_ens_loop_compression_artifacts_output_hash",
            "ix_ens_loop_compression_artifacts_config_hash",
            "ix_ens_loop_compression_artifacts_input_hash",
            "ix_ens_loop_compression_artifacts_loop_id",
        ]:
            if _has_index(inspector, "ens_loop_compression_artifacts", idx_name):
                op.drop_index(idx_name, table_name="ens_loop_compression_artifacts")
        op.drop_table("ens_loop_compression_artifacts")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_step_events"):
        if _has_index(inspector, "ens_loop_step_events", "ix_ens_loop_step_events_loop_step_after"):
            op.drop_index("ix_ens_loop_step_events_loop_step_after", table_name="ens_loop_step_events")
        if _has_index(inspector, "ens_loop_step_events", "ix_ens_loop_step_events_compression_artifact_id"):
            op.drop_index("ix_ens_loop_step_events_compression_artifact_id", table_name="ens_loop_step_events")
        if _has_column(inspector, "ens_loop_step_events", "compression_artifact_id"):
            op.drop_column("ens_loop_step_events", "compression_artifact_id")
        if _has_column(inspector, "ens_loop_step_events", "memory_payload_json"):
            op.drop_column("ens_loop_step_events", "memory_payload_json")

    inspector = sa.inspect(bind)
    if _has_table(inspector, "ens_loop_sessions"):
        if _has_index(inspector, "ens_loop_sessions", "ix_ens_loop_sessions_last_compressed_step_index"):
            op.drop_index("ix_ens_loop_sessions_last_compressed_step_index", table_name="ens_loop_sessions")
        if _has_column(inspector, "ens_loop_sessions", "last_compressed_step_index"):
            op.drop_column("ens_loop_sessions", "last_compressed_step_index")
